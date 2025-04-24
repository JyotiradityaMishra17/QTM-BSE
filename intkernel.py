
from functools import lru_cache
from typing import List, NamedTuple
import numpy as np
from scipy.spatial import Delaunay


from qtm.constants import ELECTRONVOLT_RYD
from qtm.constants import ELECTRONVOLT_HART
from qtm.crystal import Crystal
from qtm.klist import KList
from qtm.gspace.gspc import GSpace
from qtm.gspace.gkspc import GkSpace
from qtm.dft.kswfn import KSWfn

from qtm.mpi.comm import MPI4PY_INSTALLED
from kernel import KernelMtxEl
from qtm.gw.vcoul import Vcoul

from qtm.interfaces.bgw.wfn2py import WfnData
from qtm.interfaces.bgw.h5_utils import *
from qtm.gw.core import (
    QPoints,
    sort_cryst_like_BGW,
    reorder_2d_matrix_sorted_gvecs,
)

if MPI4PY_INSTALLED:
    from mpi4py import MPI

def closestpts(fine_kpt, coarse_kpts, periodic=True):
    if periodic:
        shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T

        tiled_kpts = []
        tiled_indices = []

        for shift in shifts:
            shifted = (coarse_kpts + shift) % 1.0  # wrap into unit cell
            tiled_kpts.append(shifted)
            tiled_indices.append(np.arange(len(coarse_kpts)))  # map back to original

        tiled_kpts = np.vstack(tiled_kpts)
        tiled_indices = np.hstack(tiled_indices)

        tri = Delaunay(tiled_kpts)
        simplex = tri.find_simplex(fine_kpt)

        if simplex == -1:
            raise RuntimeError("Periodic tiling failed to cover the unit cell.")

        T = tri.transform[simplex]
        X = fine_kpt - T[3]
        lam = T[:3].dot(X)
        weights = np.append(lam, 1 - lam.sum())
        vertex_ids = tri.simplices[simplex]
        indices = tiled_indices[vertex_ids]

    else:
        tri = Delaunay(coarse_kpts)
        simplex = tri.find_simplex(fine_kpt)

        if simplex == -1:
            d2 = np.sum((coarse_kpts - fine_kpt) ** 2, axis=1)
            idx = np.argmin(d2)
            return np.array([idx]), np.array([1.0])

        T = tri.transform[simplex]
        X = fine_kpt - T[3]
        lam = T[:3].dot(X)
        weights = np.append(lam, 1 - lam.sum())
        indices = tri.simplices[simplex]

    return indices, weights    


class InterpMtxEl:
    TOLERANCE = 1e-5
    ryd = 1 / ELECTRONVOLT_RYD

    def __init__(
        self,
        crystal: Crystal,
        gspace: GSpace,
        fine_kpts: KList,
        coarse_kpts: KList,
        fine_l_wfn: List[KSWfn],
        coarse_l_wfn: List[KSWfn],
        fine_l_gsp_wfn: List[GkSpace],
        coarse_l_gsp_wfn: List[GkSpace],
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        kernel: KernelMtxEl,
        periodic: bool = False,
        parallel: bool = True,
        num_val_bands_fine: int = None,
        num_con_bands_fine: int = None,
        num_val_bands_coarse: int = None,
        num_con_bands_coarse: int = None,
    ):
        self.crystal = crystal
        self.gspace = gspace
        self.fine_kpts = fine_kpts
        self.coarse_kpts = coarse_kpts

        self.fine_l_wfn = fine_l_wfn
        self.coarse_l_wfn = coarse_l_wfn
        self.fine_l_gsp_wfn = fine_l_gsp_wfn
        self.coarse_l_gsp_wfn = coarse_l_gsp_wfn

        self.epsinp = epsinp
        self.sigmainp = sigmainp

        self.kernel = kernel
        self.periodic = periodic

        self.in_parallel = False
        self.comm = None
        self.comm_size = None
        if parallel and MPI4PY_INSTALLED:
            self.comm = MPI.COMM_WORLD
            self.comm_size = self.comm.Get_size()
            if self.comm_size > 1:
                self.in_parallel = True

        self.num_bands = self.sigmainp.number_bands
        self.min_band = self.sigmainp.band_index_min
        self.max_band = self.sigmainp.band_index_max                

        occ_fine = []
        occ_coarse = []
        for k_idx in range(self.fine_kpts.numk):
            occ_fine.append(self.fine_l_wfn[k_idx].occ)
        for k_idx in range(self.coarse_kpts.numk):
            occ_coarse.append(self.coarse_l_wfn[k_idx].occ)

        occ_fine = np.array(occ_fine)
        occ_coarse = np.array(occ_coarse)

        self.occ_fine = occ_fine[:, : self.sigmainp.number_bands]
        self.occ_coarse = occ_coarse[:, : self.sigmainp.number_bands]

        self.list_con_idx_fine = np.where(self.occ_fine == 0)
        self.list_con_idx_coarse = np.where(self.occ_coarse == 0)

        self.con_idx_beg_fine = min(self.list_con_idx_fine[1])
        self.con_idx_beg_coarse = min(self.list_con_idx_coarse[1])

        self.con_num_max_fine = self.max_band - self.con_idx_beg_fine
        self.con_num_max_coarse = self.max_band - self.con_idx_beg_coarse

        # NOTE: The indexing for valence bands is different. The 1st band is the closest to fermi energy
        # and so has the highest energy. Now, our first band 0, is the furthest away - and it technically
        # should be the nth band. So we find the nth band and subtract everything from it.
        # i.e., n - n -> 0, n - 0 -> n.        

        list_val_idx_fine = np.where(self.occ_fine == 1)
        list_val_idx_coarse = np.where(self.occ_coarse == 1)

        self.val_num_max_fine = max(list_val_idx_fine[1])
        self.val_num_max_coarse = max(list_val_idx_coarse[1])

        k_idx_val_fine, band_idx_val_fine = list_val_idx_fine
        k_idx_val_coarse, band_idx_val_coarse = list_val_idx_coarse

        band_idx_val_fine = self.val_num_max_fine - band_idx_val_fine
        band_idx_val_coarse = self.val_num_max_coarse - band_idx_val_coarse

        list_val_idx_fine = [k_idx_val_fine, band_idx_val_fine]
        list_val_idx_coarse = [k_idx_val_coarse, band_idx_val_coarse]

        self.list_val_idx_fine = list_val_idx_fine
        self.list_val_idx_coarse = list_val_idx_coarse

        self.val_idx_beg_fine = min(self.list_val_idx_fine[1])
        self.val_idx_beg_coarse = min(self.list_val_idx_coarse[1])

        if num_val_bands_fine is not None:
            if num_val_bands_fine > self.val_num_max_fine + 1:
                raise ValueError(
                    f"num_val_bands_fine {num_val_bands_fine} exceeds max {self.val_num_max_fine}."
                )
            self.val_num_fine = num_val_bands_fine
        else:
            self.val_num_fine = self.val_num_max_fine + 1

        if num_val_bands_coarse is not None:
            if num_val_bands_coarse > self.val_num_max_coarse + 1:
                raise ValueError(
                    f"num_val_bands_coarse {num_val_bands_coarse} exceeds max {self.val_num_max_coarse}."
                )
            self.val_num_coarse = num_val_bands_coarse
        else:
            self.val_num_coarse = self.val_num_max_coarse + 1

        if num_con_bands_fine is not None:
            if num_con_bands_fine > self.con_num_max_fine:
                raise ValueError(
                    f"num_con_bands_fine {num_con_bands_fine} exceeds max {self.con_num_max_fine}."
                )
            self.con_num_fine = num_con_bands_fine
        else:
            self.con_num_fine = self.con_num_max_fine

        if num_con_bands_coarse is not None:
            if num_con_bands_coarse > self.con_num_max_coarse:
                raise ValueError(
                    f"num_con_bands_coarse {num_con_bands_coarse} exceeds max {self.con_num_max_coarse}."
                )
            self.con_num_coarse = num_con_bands_coarse
        else:
            self.con_num_coarse = self.con_num_max_coarse   

        l_closest_idx, l_weights = self.store_delaunay()
        self.l_closest_idx = l_closest_idx
        self.l_weights = l_weights 

        self.kernelfactor = 8 * np.pi * InterpMtxEl.ryd / (self.crystal.reallat.cellvol * self.coarse_kpts.numk)


    
    @classmethod
    def from_BGW(
        cls,
        wfn_finedata: WfnData,
        wfn_coarsedata: WfnData,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        kernel: KernelMtxEl,
        periodic: bool = False,
        parallel: bool = True,
        num_val_bands_fine: int = None,
        num_con_bands_fine: int = None,
        num_val_bands_coarse: int = None,
        num_con_bands_coarse: int = None,
    ):
        interplcass = InterpMtxEl(
            crystal=wfn_finedata.crystal,
            gspace=wfn_finedata.grho,
            fine_kpts=wfn_finedata.kpts,
            coarse_kpts=wfn_coarsedata.kpts,
            fine_l_wfn=wfn_finedata.l_wfn,
            coarse_l_wfn=wfn_coarsedata.l_wfn,
            fine_l_gsp_wfn=wfn_finedata.l_gk,
            coarse_l_gsp_wfn=wfn_coarsedata.l_gk,
            epsinp=epsinp,
            sigmainp=sigmainp,
            kernel=kernel,
            periodic=periodic,
            parallel=parallel,
            num_val_bands_fine=num_val_bands_fine,
            num_con_bands_fine=num_con_bands_fine,
            num_val_bands_coarse=num_val_bands_coarse,
            num_con_bands_coarse=num_con_bands_coarse,
        )
        
        return interplcass


    def coeff_mtxel(self, fine_k_idx: int, coarse_k_idx: int, type: str):
        if type not in ["val", "con"]:
            raise ValueError("type must be either 'val' or 'con'")

        if type == "val":
            list_idx_fine = self.list_val_idx_fine
            list_idx_coarse = self.list_val_idx_coarse

            idx_beg_coarse = self.val_idx_beg_coarse
            idx_beg_fine = self.val_idx_beg_fine

            idx_num_coarse = self.val_num_coarse
            idx_num_fine = self.val_num_fine

        else:
            list_idx_fine = self.list_con_idx_fine
            list_idx_coarse = self.list_con_idx_coarse

            idx_beg_coarse = self.con_idx_beg_coarse
            idx_beg_fine = self.con_idx_beg_fine

            idx_num_coarse = self.con_num_coarse
            idx_num_fine = self.con_num_fine

        # Final interpolation matrix shape.
        coeff_mat = np.zeros((idx_num_fine, idx_num_coarse), dtype=complex)

        @lru_cache(maxsize=int(idx_num_fine))
        def get_eigvec_fine_gk_r2g(k_idx_fine, band_idx_fine):
            band_idx = (self.val_num_max_fine - band_idx_fine) if type == "val" else band_idx_fine

            wfn_fine = self.fine_l_wfn[k_idx_fine]
            phi_fine = np.zeros(wfn_fine.gkspc.grid_shape, dtype=complex)
            self.fine_l_gsp_wfn[k_idx_fine]._fft.g2r(
                arr_inp=wfn_fine.evc_gk.data[band_idx, :],
                arr_out=phi_fine,
            )
            return phi_fine

        @lru_cache(maxsize=int(idx_num_coarse))
        def get_eigvec_coarse_gk_r2g(k_idx_coarse, band_idx_coarse):
            band_idx = (self.val_num_max_coarse - band_idx_coarse) if type == "val" else band_idx_coarse

            wfn_coarse = self.coarse_l_wfn[k_idx_coarse]
            phi_coarse = np.zeros(wfn_coarse.gkspc.grid_shape, dtype=complex)
            self.coarse_l_gsp_wfn[k_idx_coarse]._fft.g2r(
                arr_inp=wfn_coarse.evc_gk.data[band_idx, :],
                arr_out=phi_coarse,
            )
            return phi_coarse
        
        list_idx_match_fine = np.where(list_idx_fine[0] == fine_k_idx)[0]
        if len(list_idx_match_fine) == 0:
            return coeff_mat
        
        list_idx_match_coarse = np.where(list_idx_coarse[0] == coarse_k_idx)[0]
        if len(list_idx_match_coarse) == 0:
            return coeff_mat
      
        for fine_idx in list_idx_match_fine:
            fine_band_idx = list_idx_fine[1][fine_idx]

            if fine_band_idx < idx_beg_fine or fine_band_idx >= idx_beg_fine + idx_num_fine:
                continue
            
            phi_fine = get_eigvec_fine_gk_r2g(fine_k_idx, fine_band_idx)
            for coarse_idx in list_idx_match_coarse:
                coarse_band_idx = list_idx_coarse[1][coarse_idx]

                if coarse_band_idx < idx_beg_coarse or coarse_band_idx >= idx_beg_coarse + idx_num_coarse:
                    continue
                
                phi_coarse = get_eigvec_coarse_gk_r2g(coarse_k_idx, coarse_band_idx)
                overlap = np.multiply(phi_fine, np.conj(phi_coarse))
                coeff = np.sum(overlap) * self.gspace.reallat_dv

                coeff_mat[fine_band_idx - idx_beg_fine, coarse_band_idx - idx_beg_coarse] = coeff

        norms = np.sqrt(np.sum(np.abs(coeff_mat)**2, axis=-1, keepdims=True))
        coeff_mat /= (norms + 1e-10)

        return coeff_mat
    
    def store_delaunay(self):
        l_closest_idx = []
        l_weights = []

        for ikf in range(self.fine_kpts.numk):
            kfpoint = self.fine_kpts.cryst[ikf]

            idx, weights = closestpts(kfpoint, self.coarse_kpts.cryst, periodic=self.periodic)
            l_closest_idx.append(idx)
            l_weights.append(weights)
        
        return l_closest_idx, l_weights
    

   # Interpolate the quasiparticle energies. We assume that quasiparticle energies are ordered in 
    # increasing energy of valence bands i.e., we need to flip the order.
    def interp_energy(self, coarsevals: np.ndarray, type: str, parallel: bool = True):

        if type not in ["val", "con"]:
            raise ValueError("type must be either 'val' or 'con'")
        
        numk_fine = self.fine_kpts.numk
        numk_coarse = self.coarse_kpts.numk

        if type == "val":
            list_idx_fine = self.list_val_idx_fine
            list_idx_coarse = self.list_val_idx_coarse

            idx_beg_coarse = self.val_idx_beg_coarse
            idx_beg_fine = self.val_idx_beg_fine

            idx_num_coarse = self.val_num_coarse
            idx_num_fine = self.val_num_fine


        else:
            list_idx_fine = self.list_con_idx_fine
            list_idx_coarse = self.list_con_idx_coarse

            idx_beg_coarse = self.con_idx_beg_coarse
            idx_beg_fine = self.con_idx_beg_fine

            idx_num_coarse = self.con_num_coarse
            idx_num_fine = self.con_num_fine


        # Collect the quasiparticle energies.
        energy_coarse = coarsevals[:, idx_beg_coarse:idx_beg_coarse + idx_num_coarse]
        if type == "val":
            energy_coarse = np.flip(energy_coarse, axis=-1)


        # Collect the mean field energies on the fine and coarse grids.
        mf_factor = 1 / ELECTRONVOLT_HART
        
        emf_fine = np.array([self.fine_l_wfn[ikf].evl for ikf in range(numk_fine)])
        emf_fine = emf_fine[:, idx_beg_fine:idx_beg_fine + idx_num_fine] * mf_factor

        if type == "val":
            emf_fine = np.flip(emf_fine, axis=-1)

        emf_coarse = np.array([self.coarse_l_wfn[ikc].evl for ikc in range(numk_coarse)])
        emf_coarse = emf_coarse[:, idx_beg_coarse:idx_beg_coarse + idx_num_coarse] * mf_factor
        if type == "val":
            emf_coarse = np.flip(emf_coarse, axis=-1)

        energy_fine = np.zeros((numk_fine, idx_num_fine))

        def calc_interp_energy(ikf: int):
            closest_indices = self.l_closest_idx[ikf]
            weights = self.l_weights[ikf]

            coeff_mat = []
            for ikc in closest_indices:
                coeff = self.coeff_mtxel(ikf, ikc, type)
                coeff_mat.append(coeff)

            results = []
            list_idx_match_fine = np.where(list_idx_fine[0] == ikf)[0]

            for fine_idx in list_idx_match_fine:
                correction = 0.0
                band_idx_fine = list_idx_fine[1][fine_idx]

                if band_idx_fine < idx_beg_fine or band_idx_fine >= idx_beg_fine + idx_num_fine:
                    continue

                for idx, ikc in enumerate(closest_indices):
                    list_idx_match_coarse = np.where(list_idx_coarse[0] == ikc)[0]

                    if len(list_idx_match_coarse) == 0:
                        continue

                    for coarse_idx in list_idx_match_coarse:
                        band_idx_coarse = list_idx_coarse[1][coarse_idx]

                        if band_idx_coarse < idx_beg_coarse or band_idx_coarse >= idx_beg_coarse + idx_num_coarse:
                            continue
                        
                        coeff_vals = coeff_mat[idx]
                        coeff = coeff_vals[band_idx_fine - idx_beg_fine, band_idx_coarse - idx_beg_coarse]
                        coeff2 = np.abs(coeff) ** 2

                        correction += weights[idx] * coeff2 * (energy_coarse[ikc, band_idx_coarse - idx_beg_coarse] - emf_coarse[ikc, band_idx_coarse - idx_beg_coarse])

                
                energy_fine_ikf = emf_fine[ikf, band_idx_fine - idx_beg_fine] + correction
                results.append([energy_fine_ikf, band_idx_fine - idx_beg_fine])

            return results
        
        if not (self.in_parallel and parallel):
            for ikf in range(numk_fine):
                results = calc_interp_energy(ikf)
                for energy_fine_ikf, ibf in results:
                    energy_fine[ikf, ibf] = energy_fine_ikf

        else:
            proc_rank = self.comm.Get_rank()
            proc_size = self.comm.Get_size()

            if proc_rank == 0:
                indices = np.arange(numk_fine)
                chunks = np.array_split(indices, proc_size)

            else:
                chunks = None            
            local_indices = self.comm.scatter(chunks, root=0)

            local_results = []
            for ikf in local_indices:
                results = calc_interp_energy(ikf)
                for energy_fine_ikf, ibf in results:
                    local_results.append([ikf, energy_fine_ikf, ibf])

            if proc_rank != 0:
                self.comm.send(local_results, dest=0, tag=77)
                energy_fine = self.comm.bcast(None, root=0)

            else:
                all_results = local_results
                for source in range(1, proc_size):
                    local_results = self.comm.recv(source=source, tag=77)
                    all_results.extend(local_results)

                for ikf, energy_fine_ikf, ibf in all_results:
                    energy_fine[ikf, ibf] = energy_fine_ikf
                energy_fine = self.comm.bcast(energy_fine, root=0)

        return energy_fine
    
    # Note we reuse the vcoul that we would have had from the sigma calculation - since QTM does not sort it, we have to sort it while using it here.
    def interp_kernel(self, head_mtx: np.ndarray, wings_mtx: np.ndarray, body_mtx: np.ndarray, exc_mtx: np.ndarray, vcoul: Vcoul, parallel: bool = True):
        numk_fine = self.fine_kpts.numk

        list_k_coarse = self.coarse_kpts.cryst
        list_q = self.kernel.qpts.cryst

        num_val_fine = self.val_num_fine
        num_con_fine = self.con_num_fine

        def calc_interp_kernel(ikf: int, ikpf: int):
            closest_idx_ikf = self.l_closest_idx[ikf]
            weights_ikf = self.l_weights[ikf]

            coeff_val = []
            for ikc in closest_idx_ikf:
                coeff = self.coeff_mtxel(ikf, ikc, "val")
                coeff_val.append(coeff)

            coeff_con = []
            for ikc in closest_idx_ikf:
                coeff = self.coeff_mtxel(ikf, ikc, "con")
                coeff_con.append(coeff)

            closest_idx_ikpf = self.l_closest_idx[ikpf]
            weights_ikpf = self.l_weights[ikpf]

            coeffp_val = []
            for ikpc in closest_idx_ikpf:
                coeff = self.coeff_mtxel(ikpf, ikpc, "val")
                coeffp_val.append(coeff)

            coeffp_con = []
            for ikpc in closest_idx_ikpf:
                coeff = self.coeff_mtxel(ikpf, ikpc, "con")
                coeffp_con.append(coeff)

            fine_kernel = np.zeros((num_con_fine, num_con_fine, num_val_fine, num_val_fine), dtype=complex)    

            for idx_ikc, ikc in enumerate(closest_idx_ikf):
                weight_ikc = weights_ikf[idx_ikc]
                coeff_val_ikc = coeff_val[idx_ikc] # (v, a)

                coeff_con_ikc = coeff_con[idx_ikc] # (c, b)
                kc_pt = list_k_coarse[ikc]


                for idx_ikpc, ikpc in enumerate(closest_idx_ikpf):
                    weight_ikpc = weights_ikpf[idx_ikpc]
                    coeffp_val_ikpc = coeffp_val[idx_ikpc] # (V, A)

                    coeffp_con_ikpc = coeffp_con[idx_ikpc] # (C, B)
                    kpc_pt = list_k_coarse[ikpc]

                    umklapp = -np.floor(np.around(kc_pt - kpc_pt, 5))
                    q_target = kc_pt - kpc_pt + umklapp

                    iq = np.nonzero(np.all(np.isclose(list_q, q_target, atol=self.TOLERANCE), axis=1))[0][0]
                    idx_g0 = self.kernel.l_g0[iq]

                    epsinv = self.kernel.l_epsinv[iq]
                    epsinv = epsinv[idx_g0, idx_g0]

                    # vqg from QTM has 8*pi multiplied already.
                    vqg = vcoul.vcoul[iq] / (8 * np.pi)

                    # Sort vqg, using parameters for epsinv in kernel.
                    sort_order = sort_cryst_like_BGW(self.kernel.l_gq_epsinv[iq].gk_cryst, self.kernel.l_gq_epsinv[iq].gk_norm2)

                    vqg = vqg[sort_order]
                    vqg = vqg[idx_g0]

                    # Similarly, oneoverq has 8 * pi multiplied already.
                    oneoverq = vcoul.oneoverq[iq] / (8 * np.pi)

                    # Interpolate the kernel matrix elements. Shape of each matrix is (b, B, a, A)
                    einstr = "cb, va, CB, VA, bBaA -> cCvV"
                    
                    head = head_mtx[ikc, ikpc]
                    head = np.einsum(einstr, coeff_con_ikc, np.conj(coeff_val_ikc), np.conj(coeffp_con_ikpc), coeffp_val_ikpc, head, optimize=True)
                    head = head * weight_ikc * weight_ikpc * vqg * epsinv

                    wings = wings_mtx[ikc, ikpc]
                    wings = np.einsum(einstr, coeff_con_ikc, np.conj(coeff_val_ikc), np.conj(coeffp_con_ikpc), coeffp_val_ikpc, wings, optimize=True)
                    wings = wings * weight_ikc * weight_ikpc * oneoverq

                    body = body_mtx[ikc, ikpc]
                    body = np.einsum(einstr, coeff_con_ikc, np.conj(coeff_val_ikc), np.conj(coeffp_con_ikpc), coeffp_val_ikpc, body, optimize=True)
                    body = body * weight_ikc * weight_ikpc

                    exc = exc_mtx[ikc, ikpc]
                    exc = np.einsum(einstr, coeff_con_ikc, np.conj(coeff_val_ikc), np.conj(coeffp_con_ikpc), coeffp_val_ikpc, exc, optimize=True)
                    exc = -2 * exc * weight_ikc * weight_ikpc # SINGLET

                    fine_kernel += head + wings + body + exc

            return fine_kernel * self.kernelfactor


        if not (self.in_parallel and parallel):
            fine_kernel = np.zeros((numk_fine, numk_fine, num_con_fine, num_con_fine, num_val_fine, num_val_fine), dtype=complex)
            for ikf in range(numk_fine):
                for ikpf in range(numk_fine):
                    fine_kernel[ikf, ikpf] = calc_interp_kernel(ikf, ikpf)

        else:
            proc_rank = self.comm.Get_rank()
            
            if proc_rank == 0:
                idx_kf, idx_kpf = np.meshgrid(np.arange(numk_fine), np.arange(numk_fine), indexing="ij")

                full_pairs = np.stack((idx_kf.flatten(), idx_kpf.flatten()), axis=1)
                chunks = np.array_split(full_pairs, self.comm.Get_size())

            else:
                chunks = None

            local_pairs = self.comm.scatter(chunks, root=0)
            local_results = []

            for ikf, ikpf in local_pairs:
                fine_kernel = calc_interp_kernel(ikf, ikpf)
                local_results.append([ikf, ikpf, fine_kernel])

            if proc_rank != 0:
                self.comm.send(local_results, dest=0, tag=77)
                fine_kernel = self.comm.bcast(None, root=0)
            else:
                fine_kernel = np.zeros((numk_fine, numk_fine, num_con_fine, num_con_fine, num_val_fine, num_val_fine), dtype=complex)

                for ikf, ikpf, fine_kernel_ikf_ikpf in local_results:
                    fine_kernel[ikf, ikpf] = fine_kernel_ikf_ikpf

                for source in range(1, self.comm.Get_size()):
                    remote_results = self.comm.recv(source=source, tag=77)
                    for ikf, ikpf, fine_kernel_ikf_ikpf in remote_results:
                        fine_kernel[ikf, ikpf] = fine_kernel_ikf_ikpf

                fine_kernel = self.comm.bcast(fine_kernel, root=0)

        return fine_kernel          
    
    def construct_HBSE(self, finekernel: np.ndarray, fineenergyval: np.ndarray, fineenergycon: np.ndarray):
        numk_fine = self.fine_kpts.numk
        num_val_fine = self.val_num_fine
        num_con_fine = self.con_num_fine

        mtx = np.zeros((numk_fine, numk_fine, num_con_fine, num_con_fine, num_val_fine, num_val_fine), dtype=complex)

        k_idx = np.arange(numk_fine)
        c_idx = np.arange(num_con_fine)
        v_idx = np.arange(num_val_fine)

        diagvalues = fineenergycon[:, :, np.newaxis] - fineenergyval[:, np.newaxis, :]

        mtx[
            k_idx[:, None, None],
            k_idx[:, None, None],
            c_idx[None, :, None],
            c_idx[None, :, None],
            v_idx[None, None, :],
            v_idx[None, None, :],
        ] = diagvalues

        mtx += finekernel

        # Reshape it to a 2D array.
        mtx_2D = mtx.transpose(0, 2, 4, 1, 3, 5)
        mtx_2D = mtx_2D.reshape(numk_fine * num_con_fine * num_val_fine, numk_fine * num_con_fine * num_val_fine)

        return mtx_2D
