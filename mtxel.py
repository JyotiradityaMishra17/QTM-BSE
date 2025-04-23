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
    

    def matrix_element(
            self,
            type: str,                    
            k_idx: int,
            kp_idx: int | None = None,
            ret_q: bool = False,
    ):
        assert type in ("mccp", "mvvp", "mvc")

        # ---------- locate the relevant q-point(s) ----------
        if type == "mvc":
            kp_idx = k_idx
            list_q_idx = [0]                       # q = 0 only
        else:
            k_pt, kp_pt = self.kpts.cryst[k_idx], self.kpts.cryst[kp_idx]
            umklapp = -np.floor(np.around(k_pt - kp_pt, 5))
            q_target = k_pt - kp_pt + umklapp
            list_q   = self.qpts.cryst
            list_q_idx = np.nonzero(
                np.all(np.isclose(list_q, q_target, atol=self.TOLERANCE), axis=1)
            )[0]

        # ---------- pick band lists, offsets, sizes ----------
        if type == "mccp":
            list_ket_idx = list_bra_idx = self.list_con_idx
            ket_idx_beg  = bra_idx_beg  = self.con_idx_beg
            num_ket      = num_bra      = self.con_num
            is_ket_valence = is_bra_valence = False
        elif type == "mvvp":
            list_ket_idx = list_bra_idx = self.list_val_idx
            ket_idx_beg  = bra_idx_beg  = self.val_idx_beg
            num_ket      = num_bra      = self.val_num
            is_ket_valence = is_bra_valence = True
        else:  # mvc
            list_ket_idx, ket_idx_beg, num_ket = (
                self.list_con_idx, self.con_idx_beg, self.con_num
            )
            list_bra_idx, bra_idx_beg, num_bra = (
                self.list_val_idx, self.val_idx_beg, self.val_num
            )
            is_ket_valence, is_bra_valence = False, True

        # ---------- cached real-space wave-function builder ----------
        @lru_cache(maxsize=int(self.num_bands))
        def build_psi(ik: int, ib_raw: int, is_valence: bool):
            ib_act = (self.val_num_max - ib_raw) if is_valence else ib_raw
            wfn = self.l_wfn[ik]
            psi = np.zeros(wfn.gkspc.grid_shape, dtype=complex)
            self.l_gsp_wfn[ik]._fft.g2r(
                arr_inp=wfn.evc_gk._data[ib_act, :],
                arr_out=psi,
            )
            return psi

        # ---------- loop over the matching q-points ----------
        for q_idx in list_q_idx:

            # --- choose umklapp vector & FFT driver ---
            if type != "mvc":
                q_pt   = self.qpts.cryst[q_idx]
                is_q0  = bool(self.qpts.is_q0[q_idx]) if self.qpts.is_q0 is not None else False
                kp_pt  = self.kpts.cryst[kp_idx]
                umklapp = -np.floor(np.around(kp_pt, 5)) if is_q0 else \
                        -np.floor(np.around(kp_pt + q_pt, 5))
            else:
                umklapp = np.zeros(3)

            list_g_umklapp = self.l_gq[q_idx].g_cryst - umklapp[:, None]
            idxgrid = cryst2idxgrid(self.gspace.grid_shape, list_g_umklapp.astype(int))
            fft_driver = get_fft_driver()(
                self.gspace.grid_shape, idxgrid, normalise_idft=False
            )

            numg = self.l_gq[q_idx].size_g
            M = np.zeros((num_bra, num_ket, numg), dtype=complex)
            grid_vol = np.prod(self.gspace.grid_shape)

            # --- find band entries belonging to the chosen k-points ---
            ket_mask = np.where(list_ket_idx[0] == kp_idx)[0]
            bra_mask = np.where(list_bra_idx[0] == k_idx)[0]

            for iket in ket_mask:
                ib_ket_raw = list_ket_idx[1][iket]
                if not (ket_idx_beg <= ib_ket_raw < ket_idx_beg + num_ket):
                    continue
                psi_ket = build_psi(kp_idx, ib_ket_raw, is_ket_valence)

                for ibra in bra_mask:
                    ib_bra_raw = list_bra_idx[1][ibra]
                    if not (bra_idx_beg <= ib_bra_raw < bra_idx_beg + num_bra):
                        continue
                    psi_bra = build_psi(k_idx, ib_bra_raw, is_bra_valence)

                    prod = np.conj(psi_ket) * psi_bra
                    fft_prod = np.zeros(fft_driver.idxgrid.shape, dtype=complex)
                    fft_driver.r2g(prod, fft_prod)

                    M[
                        ib_bra_raw - bra_idx_beg,
                        ib_ket_raw - ket_idx_beg,
                    ] = fft_prod / grid_vol

            # --- reorder G-axis to BerkeleyGW convention ---
            sort_order = sort_cryst_like_BGW(
                self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2
            )
            M = M[:, :, sort_order]

            return (M, q_idx) if ret_q else M