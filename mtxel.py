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

class ChargeMtxEL:
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

        mtxel = ChargeMtxEL(
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
    
    def matrix_element(self,
                    mode: str,
                    k_idx: int,
                    kp_idx: int = None,
                    ret_q: bool = False):
        
        assert mode in ("mccp", "mvvp", "mvc")

        # --- find which q‐indices to do ---
        if mode == "mvc":
            kp_idx = k_idx
            q_indices = [0]
        else:
            k_pt, kp_pt = self.kpts.cryst[k_idx], self.kpts.cryst[kp_idx]
            um0 = -np.floor(np.around(k_pt - kp_pt, 5))
            q_target = k_pt - kp_pt + um0
            all_q = self.qpts.cryst
            q_indices = np.nonzero(
                np.all(np.isclose(all_q, q_target, atol=self.TOLERANCE), axis=1)
            )[0]

        # --- pick lists, sizes, offsets ---
        if mode == "mccp":
            idx_ket_arr = idx_bra_arr = self.list_con_idx
            beg_k = beg_b = self.con_idx_beg
            n_ket = n_bra = self.con_num
        elif mode == "mvvp":
            idx_ket_arr = idx_bra_arr = self.list_val_idx
            beg_k = beg_b = self.val_idx_beg
            n_ket = n_bra = self.val_num
        else:  # mvc
            idx_ket_arr, beg_k, n_ket = self.list_con_idx, self.con_idx_beg, self.con_num
            idx_bra_arr, beg_b, n_bra = self.list_val_idx, self.val_idx_beg, self.val_num

        # --- cached real‐space builder ---
        @lru_cache(maxsize=None)
        def build_psi(ik: int, raw_ib: int, is_valence: bool):
            wfn = self.l_wfn[ik]
            # flip only if valence
            ib_act = (self.val_num_max - raw_ib) if is_valence else raw_ib
            psi = np.zeros(wfn.gkspc.grid_shape, dtype=complex)
            self.l_gsp_wfn[ik]._fft.g2r(
                arr_inp=wfn.evc_gk._data[ib_act, :],
                arr_out=psi,
            )
            return psi

        # --- loop over (usually one) q_index ---
        for q_idx in q_indices:
            # build the correct umklapp and FFT driver
            if mode != "mvc":
                qpt   = self.qpts.cryst[q_idx]
                is0   = bool(self.qpts.is_q0[q_idx]) if self.qpts.is_q0 is not None else False
                umk   = -np.floor(np.around(kp_pt, 5)) if is0 else -np.floor(np.around(kp_pt + qpt, 5))
            else:
                umk = np.zeros(3)

            g_umk   = self.l_gq[q_idx].g_cryst - umk[:, None]
            idxgrid = cryst2idxgrid(self.gspace.grid_shape, g_umk.astype(int))
            driver  = get_fft_driver()(self.gspace.grid_shape, idxgrid, normalise_idft=False)

            numg = self.l_gq[q_idx].size_g
            M    = np.zeros((n_bra, n_ket, numg), dtype=complex)
            vol  = np.prod(self.gspace.grid_shape)

            # find the matching band‐entries
            ket_inds = np.where(idx_ket_arr[0] == kp_idx)[0]
            bra_inds = np.where(idx_bra_arr[0] ==  k_idx)[0]

            for ikp in ket_inds:
                raw_k  = idx_ket_arr[1][ikp]
                if not (beg_k <= raw_k < beg_k + n_ket):
                    continue
                psi_k = build_psi(kp_idx, raw_k,
                                is_valence=(mode == "mvvp"))

                for ikb in bra_inds:
                    raw_b = idx_bra_arr[1][ikb]
                    if not (beg_b <= raw_b < beg_b + n_bra):
                        continue
                    psi_b = build_psi(k_idx, raw_b,
                                    is_valence=(mode != "mccp"))

                    prod     = np.conj(psi_k) * psi_b
                    fft_prod = np.zeros(driver.idxgrid.shape, dtype=complex)
                    driver.r2g(prod, fft_prod)

                    M[raw_b - beg_b, raw_k - beg_k] = fft_prod / vol

            # reorder G‐axis to BGW convention
            sort_order = sort_cryst_like_BGW(
                self.l_gq[q_idx].gk_cryst,
                self.l_gq[q_idx].gk_norm2
            )
            M = M[:, :, sort_order]

            return (M, q_idx) if ret_q else M
    
    # mccp -> <ck|exp(i(k-kp-G0+G).r)|ckp>
    def mccp(
            self,
            k_idx: int,
            kp_idx: int,
            ret_q: bool = False,
    ):
        k_pt = self.kpts.cryst[k_idx]
        kp_pt = self.kpts.cryst[kp_idx]

        list_con_idx = deepcopy(self.list_con_idx)
        con_idx_beg = self.con_idx_beg
        num_con = self.con_num        

        list_q = self.qpts.cryst
        is_q0 = self.qpts.is_q0
        grid_vol = np.prod(self.gspace.grid_shape)

        # We need to find the index of q, for which q + G0 = k - kp.
        umklapp = -np.floor(np.around(
            k_pt - kp_pt, 5
        ))

        q_pt = k_pt - kp_pt + umklapp
        list_q_match = np.nonzero(
            np.all(
                np.isclose(
                    list_q, q_pt, atol = self.TOLERANCE
                ),
                axis = 1,
            )
        )[0]

        for q_idx in list_q_match:
            # Compute <ck|exp(i(q+G).r)|ckp>
            qpt = list_q[q_idx]
            is_qpt_0 = None if is_q0 == None else is_q0[q_idx]

            if is_qpt_0:
                umklapp = -np.floor(np.around(kp_pt, 5))
            else:
                umklapp = -np.floor(np.around(kp_pt + qpt, 5))

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


            numg = self.l_gq[q_idx].size_g
            M = np.zeros((num_con, num_con, numg), dtype=complex)

            def eigvec_ket(ik, ib):
                wfn_ket = self.l_wfn[ik]
                psi_ket = np.zeros(
                    wfn_ket.gkspc.grid_shape, dtype=complex
                )

                self.l_gsp_wfn[ik]._fft.g2r(
                    arr_inp=wfn_ket.evc_gk._data[ib, :],
                    arr_out=psi_ket,
                )

                return psi_ket
            
            def eigvec_bra(ik, ib):
                wfn_bra = self.l_wfn[ik]
                gkspc = self.l_gsp_wfn[ik]

                psi_bra = np.zeros(
                    wfn_bra.gkspc.grid_shape, dtype=complex
                )
                gkspc._fft.g2r(
                    arr_inp=wfn_bra.evc_gk._data[ib, :],
                    arr_out=psi_bra,
                )
                return psi_bra
            
            list_idxp_match_con = np.where(
                list_con_idx[0][:] == kp_idx
            )[0]
            list_idx_match_con = np.where(
                list_con_idx[0][:] == k_idx
            )[0]

            for idxp in list_idxp_match_con:
                band_idxp_con = list_con_idx[1][idxp]

                if band_idxp_con < con_idx_beg or band_idxp_con >= num_con + con_idx_beg:
                    continue

                psi_ket = eigvec_ket(kp_idx, band_idxp_con)

                for idx in list_idx_match_con:
                    band_idx_con = list_con_idx[1][idx]

                    if band_idx_con < con_idx_beg or band_idx_con >= num_con + con_idx_beg:
                        continue

                    psi_bra = eigvec_bra(k_idx, band_idx_con)
                    prod = np.multiply(np.conj(psi_ket), psi_bra)

                    fft_prod = np.zeros(
                        umklapped_fft_driver.idxgrid.shape, dtype=complex
                    )

                    umklapped_fft_driver.r2g(prod, fft_prod)

                    M[
                        band_idx_con - con_idx_beg,
                        band_idxp_con - con_idx_beg,
                    ] = fft_prod / grid_vol
            
            
            # Sort the matrix elements to match BGW.
            sort_order = sort_cryst_like_BGW(
                self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2
            )

            M = M[:, :, sort_order]
            
            if ret_q:
                return M, q_idx
            else:
                return M
            
    # mvvp -> <vk|exp(i(k-kp-G0+G).r)|vkp>
    def mvvp(
            self,
            k_idx: int,
            kp_idx: int,
            ret_q: bool = False,
    ):
        k_pt = self.kpts.cryst[k_idx]
        kp_pt = self.kpts.cryst[kp_idx]

        list_val_idx = deepcopy(self.list_val_idx)
        val_idx_beg = self.val_idx_beg
        num_val = self.val_num        

        list_q = self.qpts.cryst
        is_q0 = self.qpts.is_q0
        grid_vol = np.prod(self.gspace.grid_shape)

        # We need to find the index of q, for which q + G0 = k - kp.
        umklapp = -np.floor(np.around(
            k_pt - kp_pt, 5
        ))

        q_pt = k_pt - kp_pt + umklapp
        list_q_match = np.nonzero(
            np.all(
                np.isclose(
                    list_q, q_pt, atol = self.TOLERANCE
                ),
                axis = 1,
            )
        )[0]

        for q_idx in list_q_match:
            # Compute <vk|exp(i(q+G).r)|vkp>
            qpt = list_q[q_idx]
            is_qpt_0 = None if is_q0 == None else is_q0[q_idx]

            if is_qpt_0:
                umklapp = -np.floor(np.around(kp_pt, 5))
            else:
                umklapp = -np.floor(np.around(kp_pt + qpt, 5))

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

            numg = self.l_gq[q_idx].size_g
            M = np.zeros((num_val, num_val, numg), dtype=complex)

            def eigvec_ket(ik, ib):
                # We need to switch valence band indices back to normal ordering.
                ib_actual = self.val_num_max - ib
                wfn_ket = self.l_wfn[ik]
                psi_ket = np.zeros(
                    wfn_ket.gkspc.grid_shape, dtype=complex
                )

                self.l_gsp_wfn[ik]._fft.g2r(
                    arr_inp=wfn_ket.evc_gk._data[ib_actual, :],
                    arr_out=psi_ket,
                )

                return psi_ket
            
            def eigvec_bra(ik, ib):
                ib_actual = self.val_num_max - ib
                wfn_bra = self.l_wfn[ik]
                gkspc = self.l_gsp_wfn[ik]

                psi_bra = np.zeros(
                    wfn_bra.gkspc.grid_shape, dtype=complex
                )
                gkspc._fft.g2r(
                    arr_inp=wfn_bra.evc_gk._data[ib_actual, :],
                    arr_out=psi_bra,
                )
                return psi_bra
            
            list_idxp_match_val = np.where(
                list_val_idx[0][:] == kp_idx
            )[0]
            list_idx_match_val = np.where(
                list_val_idx[0][:] == k_idx
            )[0]

            for idxp in list_idxp_match_val:
                band_idxp_val = list_val_idx[1][idxp]

                if band_idxp_val < val_idx_beg or band_idxp_val >= num_val + val_idx_beg:
                    continue

                psi_ket = eigvec_ket(kp_idx, band_idxp_val)
     
                for idx in list_idx_match_val:
                    band_idx_val = list_val_idx[1][idx]

                    if band_idx_val < val_idx_beg or band_idx_val >= num_val + val_idx_beg:
                        continue

                    psi_bra = eigvec_bra(k_idx, band_idx_val)

                    prod = np.multiply(np.conj(psi_ket), psi_bra)

                    fft_prod = np.zeros(
                        umklapped_fft_driver.idxgrid.shape, dtype=complex
                    )

                    umklapped_fft_driver.r2g(prod, fft_prod)

                    M[
                        band_idx_val - val_idx_beg,
                        band_idxp_val - val_idx_beg,
                    ] = fft_prod / grid_vol
            
            # Sort the matrix elements to match BGW.
            sort_order = sort_cryst_like_BGW(
                self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2
            )

            M = M[:, :, sort_order]

            if ret_q:
                return M, q_idx
            else:
                return M

    # mvc -> <vk|exp(i * (G. r))|ck>
    def mvc(
            self,
            k_idx: int,
    ):
        k_pt = self.kpts.cryst[k_idx]

        list_val_idx = deepcopy(self.list_val_idx)
        list_con_idx = deepcopy(self.list_con_idx)

        val_idx_beg = self.val_idx_beg
        con_idx_beg = self.con_idx_beg

        num_val = self.val_num
        num_con = self.con_num

        # Since we have the same k for ket and bra, q is 0.
        numg = self.l_gq[0].size_g 
        list_g = self.l_gq[0].g_cryst

        idxgrid = cryst2idxgrid(
            shape = self.gspace.grid_shape,
            g_cryst = list_g.astype(int),
        )
        grid_vol = np.prod(self.gspace.grid_shape)

        fft_driver = get_fft_driver()(
            self.gspace.grid_shape,
            idxgrid,
            normalise_idft = False,
        )

        M = np.zeros((num_val, num_con, numg), dtype = complex)

        @lru_cache(maxsize=int(num_val))
        def eigvec_val(ik, ib):
            ib_actual = self.val_num_max - ib
            wfn_val = self.l_wfn[ik]
            psi_val = np.zeros(
                wfn_val.gkspc.grid_shape, dtype=complex
            )

            self.l_gsp_wfn[ik]._fft.g2r(
                arr_inp = wfn_val.evc_gk._data[ib_actual, :],
                arr_out = psi_val,
            )

            return psi_val
        
        @lru_cache(maxsize=int(num_con))
        def eigvec_con(ik, ib):

            wfn_con = self.l_wfn[ik]
            psi_con = np.zeros(
                wfn_con.gkspc.grid_shape, dtype=complex
            )

            self.l_gsp_wfn[ik]._fft.g2r(
                arr_inp = wfn_con.evc_gk._data[ib, :],
                arr_out = psi_con,
            )

            return psi_con
        
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
                    fft_driver.idxgrid.shape, dtype = complex
                )

                fft_driver.r2g(prod, fft_prod)
                M[
                    band_idx_val - val_idx_beg,
                    band_idx_con - con_idx_beg,
                ] = fft_prod / grid_vol

        # Sort the matrix elements to match BGW.
        sort_order = sort_cryst_like_BGW(
            self.l_gq[0].gk_cryst, self.l_gq[0].gk_norm2
        )

        M = M[:, :, sort_order]

        return M
    

    ####################################################
    # Original code, written separately for bug-fixing # 
    
    # def calc_charge_mtxel(
    #         self,
    #         k_idx: int,
    #         kp_idx: int = None,
    #         type: str  = None,
    #         ret_q: bool = False,
    # ):
    #     k_pt = self.kpts.cryst[k_idx]

    #     list_val_idx = deepcopy(self.list_val_idx)
    #     list_con_idx = deepcopy(self.list_con_idx)

    #     val_idx_beg = self.val_idx_beg
    #     con_idx_beg = self.con_idx_beg

    #     num_val = self.val_num
    #     num_con = self.con_num

    #     calctype = None
    #     if kp_idx is None:
    #         # We calculate the matrix elements mvc -> <vk|exp(i * (G. r))|ck>
    #         calctype = 'mvc'
    #     else:
    #         # Depending on the type, we calculate:

    #         # (a) mvvp -> <vk|exp(i(k-kp-G0+G).r)|vkp>
    #         if type == 'val':
    #             calctype = 'mvvp'

    #         # (b) mccp -> <ck|exp(i(k-kp-G0+G).r)|ckp>
    #         elif type == 'con':
    #             calctype = 'mccp'
    #         else:
    #             raise ValueError(
    #                 f"Unknown type {type}. Use 'val' or 'con'."
    #             )
            
    #     # Start calculation for type "mvc".
    #     if calctype == 'mvc':

    #         # Since we have the same k for ket and bra, q is 0.
    #         numg = self.l_gq[0].size_g 
    #         list_g = self.l_gq[0].g_cryst

    #         idxgrid = cryst2idxgrid(
    #             shape = self.gspace.grid_shape,
    #             g_cryst = list_g.astype(int),
    #         )
    #         grid_vol = np.prod(self.gspace.grid_shape)

    #         fft_driver = get_fft_driver()(
    #             self.gspace.grid_shape,
    #             idxgrid,
    #             normalise_idft = False,
    #         )

    #         M = np.zeros((num_val, num_con, numg), dtype = complex)

    #         @lru_cache(maxsize=int(num_val))
    #         def eigvec_val(ik, ib):

    #             wfn_val = self.l_wfn[ik]
    #             psi_val = np.zeros(
    #                 wfn_val.gkspc.grid_shape, dtype=complex
    #             )

    #             self.l_gsp_wfn[ik]._fft.g2r(
    #                 arr_inp = wfn_val.evc_gk._data[ib, :],
    #                 arr_out = psi_val,
    #             )

    #             return psi_val
            
    #         @lru_cache(maxsize=int(num_con))
    #         def eigvec_con(ik, ib):

    #             wfn_con = self.l_wfn[ik]
    #             psi_con = np.zeros(
    #                 wfn_con.gkspc.grid_shape, dtype=complex
    #             )

    #             self.l_gsp_wfn[ik]._fft.g2r(
    #                 arr_inp = wfn_con.evc_gk._data[ib, :],
    #                 arr_out = psi_con,
    #             )

    #             return psi_con
            
    #         list_idx_match_con = np.where(
    #             list_con_idx[0][:] == k_idx
    #         )[0]

    #         list_idx_match_val = np.where(
    #             list_val_idx[0][:] == k_idx
    #         )[0]

    #         for idx_con in list_idx_match_con:
    #             band_idx_con = list_con_idx[1][idx_con]

    #             if band_idx_con < con_idx_beg or band_idx_con >= num_con + con_idx_beg:
    #                 continue
                
    #             psi_con = eigvec_con(k_idx, band_idx_con)

    #             for idx_val in list_idx_match_val:
    #                 band_idx_val = list_val_idx[1][idx_val]

    #                 if band_idx_val < val_idx_beg or band_idx_val >= num_val + val_idx_beg:
    #                     continue

    #                 psi_val = eigvec_val(k_idx, band_idx_val)
    #                 prod = np.multiply(np.conj(psi_val), psi_con)

    #                 fft_prod = np.zeros(
    #                     fft_driver.idxgrid.shape, dtype = complex
    #                 )

    #                 fft_driver.r2g(prod, fft_prod)
    #                 M[
    #                     band_idx_val - val_idx_beg,
    #                     band_idx_con - con_idx_beg,
    #                 ] = fft_prod / grid_vol

    #         return M
        
    #     # Start calculation for type "mvvp".
    #     elif calctype == 'mvvp':
    #         kp_pt = self.kpts.cryst[kp_idx]
    #         list_q = self.qpts.cryst
    #         is_q0 = self.qpts.is_q0

    #         grid_vol = np.prod(self.gspace.grid_shape)

    #         # We need to find the index of q, for which q + G0 = k - kp.
    #         umklapp = -np.floor(np.around(
    #             k_pt - kp_pt, 5
    #         ))

    #         q_pt = k_pt - kp_pt + umklapp
    #         list_q_match = np.nonzero(
    #             np.all(
    #                 np.isclose(
    #                     list_q, q_pt, atol = self.TOLERANCE
    #                 ),
    #                 axis = 1,
    #             )
    #         )[0]

    #         for q_idx in list_q_match:

    #             # Compute <vk|exp(i(q+G).r)|vkp>
    #             qpt = list_q[q_idx]
    #             is_qpt_0 = None if is_q0 == None else is_q0[q_idx]
    #             if is_qpt_0:
    #                 umklapp = -np.floor(np.around(kp_pt, 5))
    #             else:
    #                 umklapp = -np.floor(np.around(kp_pt + qpt, 5))

    #             list_g_umklapp = self.l_gq[q_idx].g_cryst - umklapp[:, None]
    #             idxgrid = cryst2idxgrid(
    #                 shape = self.gspace.grid_shape,
    #                 g_cryst = list_g_umklapp.astype(int),
    #             )

    #             umklapped_fft_driver = get_fft_driver()(
    #                 self.gspace.grid_shape,
    #                 idxgrid,
    #                 normalise_idft = False,
    #             )

    #             numg = self.l_gq[q_idx].size_g
    #             M = np.zeros((num_val, num_val, numg), dtype = complex)                

    #             def eigvec_ket(ik, ib):
    #                 wfn_ket = self.l_wfn[ik]
    #                 psi_ket = np.zeros(
    #                     wfn_ket.gkspc.grid_shape, dtype=complex
    #                 )

    #                 self.l_gsp_wfn[ik]._fft.g2r(
    #                     arr_inp = wfn_ket.evc_gk._data[ib, :],
    #                     arr_out = psi_ket,
    #                 )

    #                 return psi_ket
                
    #             def eigvec_bra(ik, ib):
    #                 wfn_bra = self.l_wfn[ik]
    #                 gkspc = self.l_gsp_wfn[ik]

    #                 psi_bra = np.zeros(
    #                     wfn_bra.gkspc.grid_shape, dtype=complex
    #                 )
    #                 gkspc._fft.g2r(
    #                     arr_inp = wfn_bra.evc_gk._data[ib, :],
    #                     arr_out = psi_bra,
    #                 )

    #                 return psi_bra
                
    #             list_idxp_match_val = np.where(
    #                 list_val_idx[0][:] == kp_idx
    #             )[0]

    #             list_idx_match_val = np.where(
    #                 list_val_idx[0][:] == k_idx
    #             )[0]

    #             for idxp in list_idxp_match_val:
    #                 band_idxp_val = list_val_idx[1][idxp]

    #                 if band_idxp_val < val_idx_beg or band_idxp_val >= num_val + val_idx_beg:
    #                     continue

    #                 psi_ket = eigvec_ket(kp_idx, band_idxp_val)

    #                 for idx in list_idx_match_val:
    #                     band_idx_val = list_val_idx[1][idx]

    #                     if band_idx_val < val_idx_beg or band_idx_val >= num_val + val_idx_beg:
    #                         continue

    #                     psi_bra = eigvec_bra(k_idx, band_idx_val)
    #                     prod = np.multiply(np.conj(psi_ket), psi_bra)

    #                     fft_prod = np.zeros(
    #                         umklapped_fft_driver.idxgrid.shape, dtype = complex
    #                     )

    #                     umklapped_fft_driver.r2g(prod, fft_prod)

    #                     M[
    #                         band_idx_val - val_idx_beg,
    #                         band_idxp_val - val_idx_beg,
    #                     ] = fft_prod / grid_vol

    #             if ret_q:
    #                 return M, q_idx
    #             else:
    #                 return M

    #     # Start calculation for type "mccp".    
    #     else:
    #         kp_pt = self.kpts.cryst[kp_idx]
    #         list_q = self.qpts.cryst
    #         is_q0 = self.qpts.is_q0

    #         grid_vol = np.prod(self.gspace.grid_shape)

    #         # We need to find the index of q, for which q + G0 = k - kp.
    #         umklapp = -np.floor(np.around(
    #             k_pt - kp_pt, 5
    #         ))

    #         q_pt = k_pt - kp_pt + umklapp
    #         list_q_match = np.nonzero(
    #             np.all(
    #                 np.isclose(
    #                     list_q, q_pt, atol = self.TOLERANCE
    #                 ),
    #                 axis = 1,
    #             )
    #         )[0]

    #         for q_idx in list_q_match:

    #             # Compute <ck|exp(i(q+G).r)|ckp>
    #             qpt = list_q[q_idx]
    #             is_qpt_0 = None if is_q0 == None else is_q0[q_idx]

    #             if is_qpt_0:
    #                 umklapp = -np.floor(np.around(kp_pt, 5))
    #             else:
    #                 umklapp = -np.floor(np.around(kp_pt + qpt, 5))

    #             list_g_umklapp = self.l_gq[q_idx].g_cryst - umklapp[:, None]
    #             idxgrid = cryst2idxgrid(
    #                 shape = self.gspace.grid_shape,
    #                 g_cryst = list_g_umklapp.astype(int),
    #             )

    #             umklapped_fft_driver = get_fft_driver()(
    #                 self.gspace.grid_shape,
    #                 idxgrid,
    #                 normalise_idft = False,
    #             )

    #             numg = self.l_gq[q_idx].size_g
    #             M = np.zeros((num_con, num_con, numg), dtype=complex)

    #             def eigvec_ket(ik, ib):
    #                 wfn_ket = self.l_wfn[ik]
    #                 psi_ket = np.zeros(
    #                     wfn_ket.gkspc.grid_shape, dtype=complex
    #                 )

    #                 self.l_gsp_wfn[ik]._fft.g2r(
    #                     arr_inp=wfn_ket.evc_gk._data[ib, :],
    #                     arr_out=psi_ket,
    #                 )

    #                 return psi_ket
                
    #             def eigvec_bra(ik, ib):
    #                 wfn_bra = self.l_wfn[ik]
    #                 gkspc = self.l_gsp_wfn[ik]

    #                 psi_bra = np.zeros(
    #                     wfn_bra.gkspc.grid_shape, dtype=complex
    #                 )
    #                 gkspc._fft.g2r(
    #                     arr_inp=wfn_bra.evc_gk._data[ib, :],
    #                     arr_out=psi_bra,
    #                 )
    #                 return psi_bra
                
    #             list_idxp_match_con = np.where(
    #                 list_con_idx[0][:] == kp_idx
    #             )[0]
    #             list_idx_match_con = np.where(
    #                 list_con_idx[0][:] == k_idx
    #             )[0]

    #             for idxp in list_idxp_match_con:
    #                 band_idxp_con = list_con_idx[1][idxp]

    #                 if band_idxp_con < con_idx_beg or band_idxp_con >= num_con + con_idx_beg:
    #                     continue

    #                 psi_ket = eigvec_ket(kp_idx, band_idxp_con)

    #                 for idx in list_idx_match_con:
    #                     band_idx_con = list_con_idx[1][idx]

    #                     if band_idx_con < con_idx_beg or band_idx_con >= num_con + con_idx_beg:
    #                         continue

    #                     psi_bra = eigvec_bra(k_idx, band_idx_con)
    #                     prod = np.multiply(np.conj(psi_ket), psi_bra)

    #                     fft_prod = np.zeros(
    #                         umklapped_fft_driver.idxgrid.shape, dtype=complex
    #                     )

    #                     umklapped_fft_driver.r2g(prod, fft_prod)

    #                     M[
    #                         band_idx_con - con_idx_beg,
    #                         band_idxp_con - con_idx_beg,
    #                     ] = fft_prod / grid_vol
                
    #             if ret_q:
    #                 return M, q_idx
    #             else:
    #                 return M

        
          

        