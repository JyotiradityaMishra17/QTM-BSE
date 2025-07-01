# NOTE: Need to fix constants for the absorption graphs.

import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import List, NamedTuple
from copy import deepcopy

from qtm.constants import RYDBERG_HART
from qtm.crystal import Crystal
from qtm.klist import KList
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import GSpace
from qtm.dft.kswfn import KSWfn
from qtm.gw.core import QPoints, sort_cryst_like_BGW
from qtm.gspace.base import cryst2idxgrid
from qtm.fft.backend.utils import get_fft_driver

from qtm.interfaces.bgw.wfn2py import WfnData
from qtm.interfaces.bgw.h5_utils import *

from qtm.mpi.comm import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py import MPI

class Absorption:
    '''
    Class to compute the absorption matrix elements for the Bethe-Salpeter equation (BSE).
    
    The class uses equation [46] to calculate the position matrix elements, which are then used to compute
    the velocity matrix elements [45] and finally the absorption matrix elements [6]. The equations are sourced
    from "BerkeleyGW: A Massively ..."

    '''
    fixwings = True
    TOLERANCE = 1e-5

    def __init__(
        self,
        crystal: Crystal,
        gspace: GSpace,
        kpts: KList,
        kptsq: KList,
        l_wfn: List[KSWfn],
        l_wfnq: List[KSWfn],
        l_gsp_wfn: List[GkSpace],
        l_gsp_wfnq: List[GkSpace],
        qpts: QPoints,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        polarization: np.ndarray,
        num_bands_val: int = None,
        num_bands_con: int = None,
        parallel: bool = True,

    ):
        self.crystal = crystal
        self.gspace = gspace
        self.kpts = kpts
        self.kptsq = kptsq

        self.l_wfn = l_wfn
        self.l_wfnq = l_wfnq

        self.l_gsp_wfn = l_gsp_wfn
        self.l_gsp_wfnq = l_gsp_wfnq
        
        self.epsinp = epsinp
        self.sigmainp = sigmainp
        self.qpts = qpts

        self.in_parallel = False
        self.comm = None
        self.comm_size = None
        if parallel and MPI4PY_INSTALLED:
            self.comm = MPI.COMM_WORLD
            self.comm_size = self.comm.Get_size()
            if self.comm_size > 1:
                self.in_parallel = True        

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.polarization = polarization

        self.l_gq: List[GkSpace] = []
        for i_q in range(qpts.numq):
            self.l_gq.append(
                GkSpace(
                    gwfn=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                )
            )

        # NOTE: epsinv lives in a reduced gspace, due to the epsinp cutoff.
        self.l_gq_epsinv: List[GkSpace] = []
        for i_q in range(qpts.numq):
            self.l_gq_epsinv.append(
                GkSpace(
                    gwfn=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                    ecutwfn=self.epsinp.epsilon_cutoff * RYDBERG_HART,
                )
            )

        # Store the index of G = 0 for each q-point.
        self.idxG0 = self.find_g0()

        # qval0 is a vector [qx, qy, qz]
        self.qval0 = (self.kptsq.cryst - self.kpts.cryst)[0]

        self.num_bands = self.sigmainp.number_bands
        self.min_band = self.sigmainp.band_index_min
        self.max_band = self.sigmainp.band_index_max 
        
        occ = []
        for k_idx in range(self.kpts.numk):
            occ.append(self.l_wfn[k_idx].occ)
        occ = np.array(occ)
        self.occ = occ[:, : self.sigmainp.number_bands]

        self.list_con_idx = np.where(self.occ == 0)
        self.con_idx_beg = min(self.list_con_idx[1])
        self.con_num_max = self.max_band - self.con_idx_beg


        # NOTE: The indexing for valence bands is different. The 1st band is the closest to fermi energy
        # and so has the highest energy. Now, our first band 0, is the furthest away - and it technically
        # should be the nth band. So we find the nth band and subtract everything from it.
        # i.e., n - n -> 0, n - 0 -> n.

        list_val_idx = np.where(self.occ == 1) 
        self.val_num_max = max(list_val_idx[1])
        
        k_idx_val, band_idx_val = list_val_idx
        band_idx_val = self.val_num_max - band_idx_val

        list_val_idx = [k_idx_val, band_idx_val]
        self.list_val_idx = list_val_idx
        self.val_idx_beg = min(self.list_val_idx[1])

        if num_bands_val is not None:
            if num_bands_val > self.val_num_max + 1:
                raise ValueError(
                    f"num_bands_val {num_bands_val} exceeds max {self.val_num_max}."
                )
            self.val_num = num_bands_val
        else:
            self.val_num = self.val_num_max + 1

        if num_bands_con is not None:
            if num_bands_con > self.con_num_max:
                raise ValueError(
                    f"num_bands_con {num_bands_con} exceeds max {self.con_num_max}."
                )
            self.con_num = num_bands_con
        else:
            self.con_num = self.con_num_max

    
    @classmethod
    def from_BGW(
        cls,
        wfndata: WfnData,
        wfnqdata: WfnData,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        polarization: np.ndarray,
        num_bands_val: int = None,
        num_bands_con: int = None,
        parallel: bool = True,
    ):
        l_qpts = np.array(epsinp.qpts)
        l_qpts[0] *= 0
        qpts = QPoints.from_cryst(wfndata.kpts.recilat, None, *l_qpts)

        absclass = Absorption(
            crystal = wfndata.crystal,
            gspace = wfndata.grho,
            kpts = wfndata.kpts,
            kptsq = wfnqdata.kpts,
            l_wfn = wfndata.l_wfn,
            l_wfnq = wfnqdata.l_wfn,
            l_gsp_wfn = wfndata.l_gk,
            l_gsp_wfnq = wfnqdata.l_gk,
            qpts = qpts,
            epsinp = epsinp,
            sigmainp = sigmainp,
            eigvals = eigvals,
            eigvecs = eigvecs,
            polarization = polarization,
            num_bands_val = num_bands_val,
            num_bands_con = num_bands_con,
            parallel = parallel,
        )

        return absclass    
    
    @classmethod
    def from_qtm(
        cls,
        crystal: Crystal,
        gspace: GSpace,
        kpts: KList,
        kptsq: KList,
        l_wfn_kgrp: List[List[KSWfn]],
        l_wfnq_kgrp: List[List[KSWfn]],
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        polarization: np.ndarray,
        num_bands_val: int = None,
        num_bands_con: int = None,
        parallel: bool = True,
    ):
        absclass = Absorption(
            crystal = crystal,
            gspace = gspace,
            kpts = kpts,
            kptsq = kptsq,
            l_wfn = [wfn[0] for wfn in l_wfn_kgrp],
            l_wfnq = [wfn[0] for wfn in l_wfnq_kgrp],
            l_gsp_wfn = [wfn[0].gkspc for wfn in l_wfn_kgrp],
            l_gsp_wfnq = [wfn[0].gkspc for wfn in l_wfnq_kgrp],
            qpts = QPoints.from_cryst(kpts.recilat, epsinp.is_q0, *epsinp.qpts),
            epsinp = epsinp,
            sigmainp = sigmainp,
            eigvals = eigvals,
            eigvecs = eigvecs,
            polarization = polarization,
            num_bands_val = num_bands_val,
            num_bands_con = num_bands_con,
            parallel = parallel,
        )

        return absclass         

    def find_g0(self):
        """
        Find the index of the G0 vector for a given q-point, in the BGW sorted geps space.
        """

        def where_g0(iq:int):
            # Determine the sorting order of the q + G vectors in the eps space.
            sort_order = sort_cryst_like_BGW(self.l_gq_epsinv[iq].gk_cryst, self.l_gq_epsinv[iq].gk_norm2)

            qpt = self.qpts.cryst[iq]
            sorted_gvecs = self.l_gq_epsinv[iq].gk_cryst.T[sort_order]

            idx_g0 = np.where(np.all(sorted_gvecs == qpt, axis=1))[0][0]
            return idx_g0
        
        g0 = [where_g0(iq) for iq in range(self.qpts.numq)]
        return g0    

    def position_mtxel(
            self,
            k_idx: int,
    ):
        # Calculate <v(k + q)|exp(i * (q . r))|ck> / q, for q = q0.
        k_pt = self.kpts.cryst[k_idx]        
        q_idx = self.qpts.index_q0

        q0 = self.qval0
        qval0 = np.linalg.norm(q0)
        idx_G0 = self.idxG0[q_idx]

        list_val_idx = deepcopy(self.list_val_idx)
        list_con_idx = deepcopy(self.list_con_idx)

        val_idx_beg = self.val_idx_beg
        con_idx_beg = self.con_idx_beg

        num_val = self.val_num
        num_con = self.con_num

        numg = self.l_gq[q_idx].size_g
        umklapp = -np.floor(np.around(k_pt, 5))

        list_g_umklapp = self.l_gq[q_idx].g_cryst - umklapp[:, None]
        idxgrid = cryst2idxgrid(
            shape = self.gspace.grid_shape,
            g_cryst = list_g_umklapp.astype(int),
        )

        umklapped_fft_driver = get_fft_driver()(
            self.gspace.grid_shape,
            idxgrid,
            normalise_idft = False,
        )

        M = np.zeros((num_val, num_con, numg), dtype = complex)
        grid_vol = np.prod(self.gspace.grid_shape)

        @lru_cache(maxsize=int(num_val))
        def eigvec_con(ik, ib):
            wfn_ket = self.l_wfn[ik]
            psi_ket = np.zeros(
                wfn_ket.gkspc.grid_shape, dtype=complex
            )

            self.l_gsp_wfn[ik]._fft.g2r(
                arr_inp = wfn_ket.evc_gk._data[ib, :],
                arr_out = psi_ket,
            )

            return psi_ket
        
        @lru_cache(maxsize=int(num_val))
        def eigvec_val(ik, ib_raw):
            ib = (self.val_num_max - ib_raw)

            wfn_bra = self.l_wfnq[ik]
            gkspc = self.l_gsp_wfnq[ik]
        
            psi_bra = np.zeros(
                wfn_bra.gkspc.grid_shape, dtype=complex
            )
            gkspc._fft.g2r(
                arr_inp = wfn_bra.evc_gk._data[ib, :],
                arr_out = psi_bra,
            )

            return psi_bra 
             
        list_idx_match_con = np.where(
            list_con_idx[0][:] == k_idx
        )[0]

        list_idx_match_val = np.where(
            list_val_idx[0][:] == k_idx
        )[0]        

        for idx_con in list_idx_match_con:
            band_idx_con = list_con_idx[1][idx_con]

            if band_idx_con < con_idx_beg or band_idx_con >= num_con + con_idx_beg:
                continue
            
            psi_con = eigvec_con(k_idx, band_idx_con)

            for idx_val in list_idx_match_val:
                band_idx_val = list_val_idx[1][idx_val]

                if band_idx_val < val_idx_beg or band_idx_val >= num_val + val_idx_beg:
                    continue

                psi_val = eigvec_val(k_idx, band_idx_val)
                prod = np.multiply(np.conj(psi_con), psi_val)

                fft_prod = np.zeros(
                    umklapped_fft_driver.idxgrid.shape, dtype = complex
                )

                umklapped_fft_driver.r2g(prod, fft_prod)

                M[
                    band_idx_val - val_idx_beg,
                    band_idx_con - con_idx_beg,
                ] = fft_prod / grid_vol

        arr = M[..., idx_G0]

        position_mtx = np.zeros(
            (num_val, num_con, 3), dtype = complex
        )

        position_mtx[..., 0] = -1j * arr * q0[0] / qval0**2
        position_mtx[..., 1] = -1j * arr * q0[1] / qval0**2
        position_mtx[..., 2] = -1j * arr * q0[2] / qval0**2

        return position_mtx 


    def velocity_mtxel(
            self,
    ):
        numk = self.kpts.numk
        nums = len(self.eigvals)

        num_con = self.con_num
        num_val = self.val_num

        eigvecs = self.eigvecs.reshape(numk, num_val, num_con, nums)
        velocity_mtx = np.zeros((nums, 3), dtype=complex)

        for k_idx in range(numk):
            position_mtx = self.position_mtxel(k_idx) # Shape -> (v, c, D), D = 3
            eigvecs_k = eigvecs[k_idx] # Shape -> (v, c, s)

            # Calculate the sum of -i * Omega_s * A^(s)_(vck) * <vk|r|ck> 
            einstr = "s, vcs, vcD -> sD"
            velocity_mtx += np.einsum(
                einstr,
                self.eigvals,
                eigvecs_k,
                position_mtx,
                optimize=True,
            )

        velocity_mtx = -1j * velocity_mtx
        return velocity_mtx
    

    def abs_mtxel(self):
        prefac = 16 * np.pi**2  # Modify as needed.
        polarization = self.polarization

        omega_inv = 1 / (self.eigvals**2 + 1e-10)
        velocity_mtx = self.velocity_mtxel()

        # Compute e * <0|v|S>, for all S.
        dotproduct = np.zeros(
            len(self.eigvals), dtype=complex
        )

        for i in range(3):
            dotproduct += np.multiply(
                polarization[i], velocity_mtx[:, i]
            )
        dotproduct2 = np.abs(dotproduct)**2

        # Compute the delta matrix delta(omega - Omega_S).
        delta_mtx = np.eye(len(self.eigvals))

        # Compute the imaginary part of the dielectric function.
        einstr = "S, s, sS -> S"

        abs_mtx = np.einsum(
            einstr,
            prefac * omega_inv,
            dotproduct2,
            delta_mtx,
            optimize=True,
        )

        return abs_mtx
    

    def plot_abs_broadened(self, sigma=0.15):
        abs_mtx = self.abs_mtxel()    # Assumes this returns a complex array
        weights = abs_mtx.real        # Using only the real part for this plot

        # Create a fine energy grid for the broadened plot.
        energy_min = self.eigvals.min() - 80 * sigma
        energy_max = self.eigvals.max() + 80 * sigma
        energies = np.linspace(energy_min, energy_max, 1000)

        # Define the Gaussian function for broadening
        def gaussian(x, center, sigma):
            return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        # Sum the Gaussian contributions from each eigenvalue
        broadened_signal = np.zeros_like(energies)
        for e, w in zip(self.eigvals, weights):
            broadened_signal += w * gaussian(energies, e, sigma)

        # Plot the broadened absorption data
        plt.figure(figsize=(10, 6))
        plt.plot(energies, broadened_signal, label="Gaussian Broadened Absorption")
        plt.xlim(0, 10)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Absorption Matrix Element")
        plt.title("Absorption Matrix Element with Gaussian Broadening")
        plt.legend()
        plt.grid()
        plt.show()