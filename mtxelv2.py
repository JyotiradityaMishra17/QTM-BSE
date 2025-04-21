import numpy as np
from copy import deepcopy
from functools import lru_cache
from typing import List, NamedTuple

from qtm.constants import RYDBERG_HART
from qtm.crystal import Crystal
from qtm.fft.backend.utils import get_fft_driver
from qtm.klist import KList
from qtm.gw.core import sort_cryst_like_BGW
from qtm.gspace.base import cryst2idxgrid
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import GSpace
from qtm.dft.kswfn import KSWfn
from qtm.gw.core import QPoints
from qtm.interfaces.bgw.wfn2py import WfnData
from qtm.interfaces.bgw.h5_utils import *

class ChargeMtxELv2:
    TOLERANCE = 1e-5
    vcutoff = 1e-5

    def __init__(
        self,
        crystal: Crystal,
        gspace: GSpace,
        kpts: KList,
        l_wfn: List[KSWfn],
        l_gsp_wfn: List[GkSpace],
        qpts: QPoints,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        num_bands_val: int = None,
        num_bands_con: int = None,
    ):
        self.crystal = crystal
        self.gspace = gspace
        self.kpts = kpts
        self.l_wfn = l_wfn
        self.l_gsp_wfn = l_gsp_wfn
        self.qpts = qpts
        self.epsinp = epsinp
        self.sigmainp = sigmainp

        self.l_gq: List[GkSpace] = []
        for i_q in range(qpts.numq):
            self.l_gq.append(
                GkSpace(
                    gwfn=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                )
            )

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
        self.val_num_max = max(list_val_idx[1]) + 1
        
        k_idx_val, band_idx_val = list_val_idx
        band_idx_val = self.val_num_max - band_idx_val

        list_val_idx = [k_idx_val, band_idx_val]
        self.list_val_idx = list_val_idx
        self.val_idx_beg = min(self.list_val_idx[1])

        if num_bands_val is not None:
            if num_bands_val > self.val_num_max:
                raise ValueError(
                    f"num_bands_val {num_bands_val} exceeds max {self.val_num_max}."
                )
            self.val_num = num_bands_val
        else:
            self.val_num = self.val_num_max

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
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        num_bands_val: int = None,
        num_bands_con: int = None,
    ):
        l_qpts = np.array(epsinp.qpts)
        l_qpts[0] *= 0
        qpts = QPoints.from_cryst(wfndata.kpts.recilat, None, *l_qpts)

        mtxel = ChargeMtxELv2(
            crystal = wfndata.crystal,
            gspace = wfndata.grho,
            kpts = wfndata.kpts,
            l_wfn = wfndata.l_wfn,
            l_gsp_wfn = wfndata.l_gk,
            qpts = qpts,
            epsinp = epsinp,
            sigmainp = sigmainp,
            num_bands_val = num_bands_val,
            num_bands_con = num_bands_con,
        )        

        return mtxel
    
    def mtxelv2(
            self,
            q_idx: int,
            kp_idx: int,
            type: str,
            ret_k: bool = False,
    ):
        
        if type not in ["mvc", "mccp", "mvvp"]:
            raise ValueError(f"Invalid type: {type}. Must be one of ['mvc', 'mccp', 'mvvp'].")
        
        list_k = self.kpts.cryst
        kp_pt = list_k[kp_idx]

        is_q0 = self.qpts.is_q0
        list_q = self.qpts.cryst

        num_bands = self.num_bands
        num_val = self.val_num
        num_con = self.con_num

        list_val_idx = deepcopy(self.list_val_idx)
        list_con_idx = deepcopy(self.list_con_idx)

        val_idx_beg = self.val_idx_beg
        con_idx_beg = self.con_idx_beg        

        grid_vol = np.prod(self.gspace.grid_shape)
        max_cache_size = int(max(self.val_num, num_bands))

        @lru_cache(maxsize=max_cache_size)
        def get_eigvec(k_idx, band_idx):
            wfn = self.l_wfn[k_idx]
            gkspc = self.l_gsp_wfn[k_idx]

            phi = np.zeros(self.gspace.grid_shape, dtype=complex)
            gkspc._fft.g2r(wfn.evc_gk._data[band_idx, :], arr_out=phi)
            return phi


        if type == "mvc": # -> <vkp|exp(i * (G. r))|ckp>
            list_g = self.l_gq[0].g_cryst
            sort_order = sort_cryst_like_BGW(self.l_gq[0].gk_cryst, self.l_gq[0].gk_norm2)

            idxgrid = cryst2idxgrid(shape = self.gspace.grid_shape, g_cryst = list_g.astype(int))
            fft_driver = get_fft_driver()(self.gspace.grid_shape, idxgrid, normalise_idft=False)

            M = np.zeros((num_val, num_con, self.l_gq[0].size_g), dtype=complex)

            list_ket_idx = list_con_idx
            list_bra_idx = list_val_idx

            l_idx_match_ket = np.where(list_con_idx[0][:] == kp_idx)[0]
            l_idx_match_bra = np.where(list_val_idx[0][:] == kp_idx)[0]

            ket_idx_beg = con_idx_beg
            bra_idx_beg = val_idx_beg

            num_ket = num_con
            num_bra = num_val

        else:
            qpt = list_q[q_idx]
            is_qpt_0 = None if is_q0 == None else is_q0[q_idx]

            if is_qpt_0:
                umklapp = -np.floor(np.around(kp_pt, 5))
                k_pt = kp_pt + umklapp
            else:
                umklapp = -np.floor(np.around(kp_pt + qpt, 5))
                k_pt = kp_pt + qpt + umklapp

            list_g_umklapp = self.l_gq[q_idx].g_cryst - umklapp[:, None]
            sort_order = sort_cryst_like_BGW(self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2)

            idxgrid = cryst2idxgrid(shape = self.gspace.grid_shape, g_cryst = list_g_umklapp.astype(int))
            fft_driver = get_fft_driver()(self.gspace.grid_shape, idxgrid, normalise_idft=False)

            if type == "mccp": # -> <ck|exp(i(k-kp-G0+G).r)|ckp>
                M = np.zeros((num_con, num_con, self.l_gq[q_idx].size_g), dtype=complex)

                list_ket_idx = list_con_idx
                l_idx_match_ket = np.where(list_con_idx[0][:] == kp_idx)[0]
                ket_idx_beg = con_idx_beg
                num_ket = num_con

                list_bra_idx = list_con_idx
                list_k_bra = list_k[list_con_idx[0][:], :]
                l_idx_match_bra = np.nonzero(np.all(np.isclose(list_k_bra, k_pt[None, :], atol=ChargeMtxELv2.TOLERANCE), axis=1))[0]
                bra_idx_beg = con_idx_beg
                num_bra = num_con

            else: # mvvp -> <vk|exp(i(k-kp-G0+G).r)|vkp>
                M = np.zeros((num_val, num_val, self.l_gq[q_idx].size_g), dtype=complex)

                list_ket_idx = list_val_idx
                l_idx_match_ket = np.where(list_val_idx[0][:] == kp_idx)[0]
                ket_idx_beg = val_idx_beg
                num_ket = num_val

                list_bra_idx = list_val_idx
                list_k_bra = list_k[list_val_idx[0][:], :]
                l_idx_match_bra = np.nonzero(np.all(np.isclose(list_k_bra, k_pt[None, :], atol=ChargeMtxELv2.TOLERANCE), axis=1))[0]
                bra_idx_beg = val_idx_beg
                num_bra = num_val

        # Compute the matrix element.
        for ip in l_idx_match_ket:
            ikp = list_ket_idx[0][ip]
            ibp = list_ket_idx[1][ip]

            if ibp < ket_idx_beg or ibp >= ket_idx_beg + num_ket:
                continue
            
            if type == "mvvp":
                phi_ket = get_eigvec(ikp, self.val_num_max - ibp)
            else:
                phi_ket = get_eigvec(ikp, ibp)

            for i in l_idx_match_bra:
                ik = list_bra_idx[0][i]
                ib = list_bra_idx[1][i]

                if ib < bra_idx_beg or ib >= bra_idx_beg + num_bra:
                    continue
                
                if type == "mccp":
                    phi_bra = get_eigvec(ik, ib)
                else:
                    phi_bra = get_eigvec(ik, self.val_num_max - ib)

                prod = np.multiply(np.conj(phi_ket), phi_bra)

                fft_prod = np.zeros(fft_driver.idxgrid.shape, dtype=complex)
                fft_driver.r2g(prod, fft_prod)

                M[ib - bra_idx_beg, ibp - ket_idx_beg] = fft_prod / grid_vol

        # Sort the matrix elements according to BGW convention.
        M = M[:, :, sort_order]

        if ret_k:
            return M, ik
        else:
            return M




        




    

        
          

        