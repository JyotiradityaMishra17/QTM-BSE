import numpy as np
from copy import deepcopy
from functools import lru_cache
from typing import List, NamedTuple

from qtm.constants import RYDBERG_HART
from qtm.crystal import Crystal
from qtm.fft.backend.utils import get_fft_driver
from qtm.klist import KList
from qtm.gspace.base import cryst2idxgrid
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import GSpace
from qtm.dft.kswfn import KSWfn
from qtm.gw.core import QPoints

class MtxElv2:
    TOLERANCE = 1e-5
    vcutoff = 1e-5

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
        num_bands_val: int = None,
        num_bands_con: int = None,
    ):
        self.crystal = crystal
        self.gspace = gspace
        self.kpts = kpts
        self.kptsq = kptsq
        self.l_wfn = l_wfn
        self.l_wfnq = l_wfnq
        self.l_gsp_wfn = l_gsp_wfn
        self.l_gsp_wfnq = l_gsp_wfnq
        self.qpts = qpts
        self.epsinp = epsinp
        self.sigmainp = sigmainp

        self.l_gq: List[GkSpace] = []
        for i_q in range(qpts.numq):
            self.l_gq.append(
                GkSpace(
                    gwfn=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                    ecutwfn=self.epsinp.epsilon_cutoff * RYDBERG_HART,
                )
            )

        occ = []
        for k_idx in range(self.kpts.numk):
            occ.append(self.l_wfn[k_idx].occ)
        occ = np.array(occ)
        self.occ = occ[:, : self.sigmainp.number_bands]

        self.list_val_idx = np.where(self.occ == 1)
        self.list_con_idx = np.where(self.occ == 0)

        self.num_bands = self.sigmainp.number_bands
        self.min_band = self.sigmainp.band_index_min
        self.max_band = self.sigmainp.band_index_max

        self.val_idx_beg = 0
        self.con_idx_beg = min(self.list_con_idx[1])

        val_num_max = max(self.list_val_idx[1]) + 1
        con_num_max = self.max_band - self.con_idx_beg

        if num_bands_val is not None:
            if num_bands_val > val_num_max:
                raise ValueError(
                    f"num_bands_val {num_bands_val} exceeds max {val_num_max}."
                )
            self.val_num = num_bands_val
        else:
            self.val_num = val_num_max

        if num_bands_con is not None:
            if num_bands_con > con_num_max:
                raise ValueError(
                    f"num_bands_con {num_bands_con} exceeds max {con_num_max}."
                )
            self.con_num = num_bands_con
        else:
            self.con_num = con_num_max

    def mtxel_s1(self, q_idx: int, bra: str, ket: str):
        # Check if the bra and ket are valid.
        if bra not in ["val", "con"]:
            raise ValueError(f"Invalid bra: {bra}.")
        if ket not in ["val", "con"]:
            raise ValueError(f"Invalid ket: {ket}.")

        # Extract k-point, q-point, and occupation information.
        numk = self.kpts.numk
        list_k = self.kpts.cryst
        is_q0 = self.qpts.is_q0
        list_q = self.qpts.cryst
        num_bands = self.num_bands

        # Compute the umklapp.
        qpt = list_q[q_idx]
        is_qpt_0 = None if is_q0 == None else is_q0[q_idx]

        if is_qpt_0:
            umklapp = -np.floor(np.around(list_k, 5))
            list_kq = list_k + umklapp
        else:
            umklapp = -np.floor(np.around(list_k + qpt, 5))
            list_kq = list_k + qpt + umklapp

        # Define FFT functions.
        grid_vol = np.prod(self.gspace.grid_shape)
        max_cache_size = int(max(self.val_num, num_bands))

        @lru_cache(maxsize=max_cache_size)
        def get_eigvec_bra_gk_r2g(k_idx_bra, band_idx_bra):
            if is_qpt_0:
                wfn_bra = self.l_wfnq[k_idx_bra]
                gkspc = self.l_gsp_wfnq[k_idx_bra]
            else:
                wfn_bra = self.l_wfn[k_idx_bra]
                gkspc = self.l_gsp_wfn[k_idx_bra]

            arr_r = np.zeros(self.gspace.grid_shape, dtype=complex)
            gkspc._fft.g2r(
                wfn_bra.evc_gk._data[band_idx_bra, :], arr_out=arr_r
            )
            return arr_r

        @lru_cache(maxsize=max_cache_size)
        def get_eigvec_ket_gk_r2g(k_idx_ket, band_idx_ket):
            wfn_ket = self.l_wfn[k_idx_ket]
            phi_ket = np.zeros(wfn_ket.gkspc.grid_shape, dtype=complex)
            self.l_gsp_wfn[k_idx_ket]._fft.g2r(
                arr_inp=wfn_ket.evc_gk._data[band_idx_ket, :],
                arr_out=phi_ket,
            )
            return phi_ket

        # Extract band indices for matrix element computation.
        if bra == "val":
            list_bra_idx = deepcopy(self.list_val_idx)
            bra_idx_beg = self.val_idx_beg
            num_bra = self.val_num
        else:
            list_bra_idx = deepcopy(self.list_con_idx)
            bra_idx_beg = self.con_idx_beg
            num_bra = self.con_num

        if ket == "val":
            list_ket_idx = deepcopy(self.list_val_idx)
            ket_idx_beg = self.val_idx_beg
            num_ket = self.val_num
        else:
            list_ket_idx = deepcopy(self.list_con_idx)
            ket_idx_beg = self.con_idx_beg
            num_ket = self.con_num

        len_ket = len(list_ket_idx[0])
        numg = self.l_gq[q_idx].size_g

        M = np.zeros((numk, num_bra, num_ket, numg), dtype=complex)

        prev_k_idx_ket = None
        for ket_idx in range(len_ket):
            k_idx_ket = list_ket_idx[0][ket_idx]
            band_idx_ket = list_ket_idx[1][ket_idx]

            if band_idx_ket < ket_idx_beg or band_idx_ket >= num_ket + ket_idx_beg:
                continue

            phi_ket = get_eigvec_ket_gk_r2g(k_idx_ket, band_idx_ket)

            if prev_k_idx_ket != k_idx_ket:
                prev_k_idx_ket = k_idx_ket
                list_g_umklapp = (
                    self.l_gq[q_idx].g_cryst
                    - umklapp[k_idx_ket][:, None]
                )
                idxgrid = cryst2idxgrid(
                    shape=self.gspace.grid_shape,
                    g_cryst=list_g_umklapp.astype(int),
                )
                umklapped_fft_driver = get_fft_driver()(
                    self.gspace.grid_shape,
                    idxgrid,
                    normalise_idft=False,
                )

            list_k_bra = list_k[list_bra_idx[0][:], :]
            k_con_q = list_kq[k_idx_ket]
            list_idx_match = np.nonzero(
                np.all(
                    np.isclose(
                        list_k_bra,
                        k_con_q[None, :],
                        atol=MtxElv2.TOLERANCE,
                    ),
                    axis=1,
                )
            )[0]

            for idx_match in list_idx_match:
                k_idx_bra = list_bra_idx[0][idx_match]
                band_idx_bra = list_bra_idx[1][idx_match]

                if band_idx_bra < bra_idx_beg or band_idx_bra >= num_bra + bra_idx_beg:
                    continue

                phi_bra = get_eigvec_bra_gk_r2g(k_idx_bra, band_idx_bra)
                prod = np.multiply(np.conj(phi_ket), phi_bra)
                fft_prod = np.zeros(
                    umklapped_fft_driver.idxgrid.shape, dtype=complex
                )
                umklapped_fft_driver.r2g(prod, fft_prod)
                M[k_idx_ket,
                  band_idx_bra - bra_idx_beg,
                  band_idx_ket - ket_idx_beg] = fft_prod / grid_vol

        return M

    def mtxel_s2(self, q_idx: int, k_idx_ket: int, bra: str, ket: str):
        # Check if the bra and ket are valid.
        if bra not in ["val", "con"]:
            raise ValueError(f"Invalid bra: {bra}.")
        if ket not in ["val", "con"]:
            raise ValueError(f"Invalid ket: {ket}.")

        list_k = self.kpts.cryst
        is_q0 = self.qpts.is_q0
        list_q = self.qpts.cryst
        num_bands = self.num_bands

        # Compute the umklapp.
        qpt = list_q[q_idx]
        is_qpt_0 = None if is_q0 == None else is_q0[q_idx]

        kpt_ket = list_k[k_idx_ket]
        if is_qpt_0:
            umklapp = -np.floor(np.around(kpt_ket, 5))
            kqpt = kpt_ket + umklapp
        else:
            umklapp = -np.floor(np.around(kpt_ket + qpt, 5))
            kqpt = kpt_ket + qpt + umklapp

        grid_vol = np.prod(self.gspace.grid_shape)
        max_cache_size = int(max(self.val_num, num_bands))

        @lru_cache(maxsize=max_cache_size)
        def get_eigvec_bra_gk_r2g(k_idx_bra, band_idx_bra):
            if is_qpt_0:
                wfn_bra = self.l_wfnq[k_idx_bra]
                gkspc = self.l_gsp_wfnq[k_idx_bra]
            else:
                wfn_bra = self.l_wfn[k_idx_bra]
                gkspc = self.l_gsp_wfn[k_idx_bra]

            arr_r = np.zeros(self.gspace.grid_shape, dtype=complex)
            gkspc._fft.g2r(
                wfn_bra.evc_gk._data[band_idx_bra, :], arr_out=arr_r
            )
            return arr_r

        @lru_cache(maxsize=max_cache_size)
        def get_eigvec_ket_gk_r2g(k_idx_ket, band_idx_ket):
            wfn_ket = self.l_wfn[k_idx_ket]
            phi_ket = np.zeros(wfn_ket.gkspc.grid_shape, dtype=complex)
            self.l_gsp_wfn[k_idx_ket]._fft.g2r(
                arr_inp=wfn_ket.evc_gk._data[band_idx_ket, :],
                arr_out=phi_ket,
            )
            return phi_ket

        if bra == "val":
            list_bra_idx = deepcopy(self.list_val_idx)
            bra_idx_beg = self.val_idx_beg
            num_bra = self.val_num
        else:
            list_bra_idx = deepcopy(self.list_con_idx)
            bra_idx_beg = self.con_idx_beg
            num_bra = self.con_num

        if ket == "val":
            list_ket_idx = deepcopy(self.list_val_idx)
            ket_idx_beg = self.val_idx_beg
            num_ket = self.val_num
        else:
            list_ket_idx = deepcopy(self.list_con_idx)
            ket_idx_beg = self.con_idx_beg
            num_ket = self.con_num

        numg = self.l_gq[q_idx].size_g
        M = np.zeros((num_bra, num_ket, numg), dtype=complex)

        list_ket_idx_match = np.where(list_ket_idx[0] == k_idx_ket)[0]
        if len(list_ket_idx_match) == 0:
            return M

        for ket_idx in list_ket_idx_match:
            band_idx_ket = list_ket_idx[1][ket_idx]

            if band_idx_ket < ket_idx_beg or band_idx_ket >= num_ket + ket_idx_beg:
                continue

            phi_ket = get_eigvec_ket_gk_r2g(k_idx_ket, band_idx_ket)
            list_g_umklapp = self.l_gq[q_idx].g_cryst - umklapp[:, None]
            idxgrid = cryst2idxgrid(
                shape=self.gspace.grid_shape,
                g_cryst=list_g_umklapp.astype(int),
            )
            umklapped_fft_driver = get_fft_driver()(
                self.gspace.grid_shape,
                idxgrid,
                normalise_idft=False,
            )
            list_k_bra = list_k[list_bra_idx[0][:], :]
            list_idx_match = np.nonzero(
                np.all(
                    np.isclose(
                        list_k_bra, kqpt[None, :],
                        atol=MtxElv2.TOLERANCE,
                    ),
                    axis=1,
                )
            )[0]

            for idx_match in list_idx_match:
                k_idx_bra = list_bra_idx[0][idx_match]
                band_idx_bra = list_bra_idx[1][idx_match]

                if band_idx_bra < bra_idx_beg or band_idx_bra >= num_bra + bra_idx_beg:
                    continue

                phi_bra = get_eigvec_bra_gk_r2g(k_idx_bra, band_idx_bra)
                prod = np.multiply(np.conj(phi_ket), phi_bra)
                fft_prod = np.zeros(
                    umklapped_fft_driver.idxgrid.shape, dtype=complex
                )
                umklapped_fft_driver.r2g(prod, fft_prod)
                M[band_idx_bra - bra_idx_beg,
                  band_idx_ket - ket_idx_beg] = fft_prod / grid_vol

        return M
    

        




