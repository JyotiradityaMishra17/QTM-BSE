# from functools import lru_cache
# from typing import List, NamedTuple
# import numpy as np

# from qtm.constants import ELECTRONVOLT_HART
# from qtm.crystal import Crystal
# from qtm.klist import KList
# from qtm.gspace.gspc import GSpace
# from qtm.gspace.gkspc import GkSpace
# from qtm.dft.kswfn import KSWfn

# from qtm.mpi.comm import MPI4PY_INSTALLED
# from kernel import KernelMtxEl
# from qtm.interfaces.bgw.wfn2py import WfnData
# from qtm.interfaces.bgw.h5_utils import *
# from qtm.gw.core import (
#     QPoints,
#     sort_cryst_like_BGW,
#     reorder_2d_matrix_sorted_gvecs,
# )

# if MPI4PY_INSTALLED:
#     from mpi4py import MPI


# class InterpMtxEl:
#     def __init__(
#         self,
#         crystal: Crystal,
#         gspace: GSpace,
#         fine_kpts: KList,
#         coarse_kpts: KList,
#         fine_l_wfn: List[KSWfn],
#         coarse_l_wfn: List[KSWfn],
#         fine_l_gsp_wfn: List[GkSpace],
#         coarse_l_gsp_wfn: List[GkSpace],
#         kernel: KernelMtxEl,
#         parallel: bool = True,
#         num_val_bands_fine: int = None,
#         num_con_bands_fine: int = None,
#         num_val_bands_coarse: int = None,
#         num_con_bands_coarse: int = None,
#     ):
#         self.crystal = crystal
#         self.gspace = gspace
#         self.fine_kpts = fine_kpts
#         self.coarse_kpts = coarse_kpts

#         self.fine_l_wfn = fine_l_wfn
#         self.coarse_l_wfn = coarse_l_wfn
#         self.fine_l_gsp_wfn = fine_l_gsp_wfn
#         self.coarse_l_gsp_wfn = coarse_l_gsp_wfn

#         self.kernel = kernel
#         self.sigmainp = self.kernel.sigmainp
#         self.epsinp = self.kernel.epsinp

#         self.in_parallel = False
#         self.comm = None
#         self.comm_size = None
#         if parallel and MPI4PY_INSTALLED:
#             self.comm = MPI.COMM_WORLD
#             self.comm_size = self.comm.Get_size()
#             if self.comm_size > 1:
#                 self.in_parallel = True

#         occ_fine = []
#         occ_coarse = []
#         for k_idx in range(self.fine_kpts.numk):
#             occ_fine.append(self.fine_l_wfn[k_idx].occ)
#         for k_idx in range(self.coarse_kpts.numk):
#             occ_coarse.append(self.coarse_l_wfn[k_idx].occ)

#         occ_fine = np.array(occ_fine)
#         occ_coarse = np.array(occ_coarse)
#         self.occ_fine = occ_fine[:, : self.sigmainp.number_bands]
#         self.occ_coarse = occ_coarse[:, : self.sigmainp.number_bands]

#         self.num_bands = self.sigmainp.number_bands
#         self.min_band  = self.sigmainp.band_index_min
#         self.max_band  = self.sigmainp.band_index_max

#         def _compute_band_indices(occ):
#             # find conduction bands
#             list_con_idx = np.where(occ == 0)
#             con_idx_beg = list_con_idx[1].min()
#             con_num_max = self.max_band - con_idx_beg

#             # find valence bands (flip ordering so band 0 → highest energy)
#             raw_val = np.where(occ == 1)
#             val_num_max = raw_val[1].max() + 1
#             k_idx_val, b_idx_val = raw_val
#             b_idx_val = val_num_max - b_idx_val
#             list_val_idx = [k_idx_val, b_idx_val]
#             val_idx_beg = b_idx_val.min()

#             return list_val_idx, list_con_idx, val_idx_beg, con_idx_beg, val_num_max, con_num_max

#         # ──────────────── Coarse k‐points ────────────────
#         self.list_val_idx_coarse, \
#         self.list_con_idx_coarse, \
#         self.val_idx_beg_coarse, \
#         self.con_idx_beg_coarse, \
#         self.val_num_coarse_max, \
#         self.con_num_coarse_max = _compute_band_indices(self.occ_coarse)

#         if num_val_bands_coarse is not None:
#             if num_val_bands_coarse > self.val_num_coarse_max:
#                 raise ValueError(f"num_val_bands_coarse {num_val_bands_coarse} exceeds max {self.val_num_coarse_max}.")
#             self.val_num_coarse = num_val_bands_coarse
#         else:
#             self.val_num_coarse = self.val_num_coarse_max

#         if num_con_bands_coarse is not None:
#             if num_con_bands_coarse > self.con_num_coarse_max:
#                 raise ValueError(f"num_con_bands_coarse {num_con_bands_coarse} exceeds max {self.con_num_coarse_max}.")
#             self.con_num_coarse = num_con_bands_coarse
#         else:
#             self.con_num_coarse = self.con_num_coarse_max

#         # ──────────────── Fine k‐points ────────────────
#         self.list_val_idx_fine, \
#         self.list_con_idx_fine, \
#         self.val_idx_beg_fine, \
#         self.con_idx_beg_fine, \
#         self.val_num_fine_max, \
#         self.con_num_fine_max = _compute_band_indices(self.occ_fine)

#         if num_val_bands_fine is not None:
#             if num_val_bands_fine > self.val_num_fine_max:
#                 raise ValueError(f"num_val_bands_fine {num_val_bands_fine} exceeds max {self.val_num_fine_max}.")
#             self.val_num_fine = num_val_bands_fine
#         else:
#             self.val_num_fine = self.val_num_fine_max

#         if num_con_bands_fine is not None:
#             if num_con_bands_fine > self.con_num_fine_max:
#                 raise ValueError(f"num_con_bands_fine {num_con_bands_fine} exceeds max {self.con_num_fine_max}.")
#             self.con_num_fine = num_con_bands_fine
#         else:
#             self.con_num_fine = self.con_num_fine_max

    
#     @classmethod
#     def from_BGW(
#         cls,
#         wfn_finedata: WfnData,
#         wfn_coarsedata: WfnData,
#         kernel: KernelMtxEl,
#         parallel: bool = True,
#         num_val_bands_fine: int = None,
#         num_con_bands_fine: int = None,
#         num_val_bands_coarse: int = None,
#         num_con_bands_coarse: int = None,
#     ):
#         interplcass = InterpMtxEl(
#             crystal=wfn_finedata.crystal,
#             gspace=wfn_finedata.grho,
#             fine_kpts=wfn_finedata.kpts,
#             coarse_kpts=wfn_coarsedata.kpts,
#             fine_l_wfn=wfn_finedata.l_wfn,
#             coarse_l_wfn=wfn_coarsedata.l_wfn,
#             fine_l_gsp_wfn=wfn_finedata.l_gk,
#             coarse_l_gsp_wfn=wfn_coarsedata.l_gk,
#             kernel=kernel,
#             parallel=parallel,
#             num_val_bands_fine=num_val_bands_fine,
#             num_con_bands_fine=num_con_bands_fine,
#             num_val_bands_coarse=num_val_bands_coarse,
#             num_con_bands_coarse=num_con_bands_coarse,
#         )
        
#         return interplcass

#     def coeff(self, ikf: int, ikc: int, type: str):
#         if type not in ["val", "con"]:
#             raise ValueError(f"Invalid type {type}. Must be 'val' or 'con'.")
        
#         if type == "val":
#             list_idx_fine = self.list_val_idx_fine
#             list_idx_coarse = self.list_val_idx_coarse

#             idx_beg_fine = self.val_idx_beg_fine
#             idx_beg_coarse = self.val_idx_beg_coarse

#             idx_num_fine = self.val_num_fine
#             idx_num_coarse = self.val_num_coarse

#             isvalflag = True
#         else:
#             list_idx_fine = self.list_con_idx_fine
#             list_idx_coarse = self.list_con_idx_coarse

#             idx_beg_fine = self.con_idx_beg_fine
#             idx_beg_coarse = self.con_idx_beg_coarse

#             idx_num_fine = self.con_num_fine
#             idx_num_coarse = self.con_num_coarse

#             isvalflag = False

#         # Allocate matrix.
#         cmtx = np.zeros((idx_num_fine, idx_num_coarse), dtype=complex)

#         @lru_cache(maxsize=int(idx_num_fine))
#         def get_psi_fine(ik, ib, isvalflag):
#             ib_actual = self.val_num_fine_max - ib if isvalflag else ib
#             wfn_fine = self.fine_l_wfn[ik]

#             psi_fine = np.zeros(wfn_fine.gkspc.grid_shape, dtype=complex)
#             self.fine_l_gsp_wfn[ik]._fft.g2r(
#                 arr_inp = wfn_fine.evc_gk.data[ib_actual, :],
#                 arr_out = psi_fine,
#             )

#             return psi_fine
        
#         @lru_cache(maxsize=int(idx_num_coarse))
#         def get_psi_coarse(ik, ib, isvalflag):
#             ib_actual = self.val_num_coarse_max - ib if isvalflag else ib
#             wfn_coarse = self.coarse_l_wfn[ik]

#             psi_coarse = np.zeros(wfn_coarse.gkspc.grid_shape, dtype=complex)
#             self.coarse_l_gsp_wfn[ik]._fft.g2r(
#                 arr_inp = wfn_coarse.evc_gk.data[ib_actual, :],
#                 arr_out = psi_coarse,
#             )

#             return psi_coarse
        
#         # Loop over band indices.
#         list_idx_match_fine = np.where(list_idx_fine[0] == ikf)[0]
#         if len(list_idx_match_fine) == 0:
#             return cmtx
        
#         list_idx_match_coarse = np.where(list_idx_coarse[0] == ikc)[0]
#         if len(list_idx_match_coarse) == 0:
#             return cmtx
        
#         for idx_fine in list_idx_match_fine:
#             ibf = list_idx_fine[1][idx_fine]
#             if ibf < idx_beg_fine or ibf >= idx_beg_fine + idx_num_fine:
#                 continue

#             phi_fine = get_psi_fine(ikf, ibf, isvalflag)
#             for idx_coarse in list_idx_match_coarse:
#                 ibc = list_idx_coarse[1][idx_coarse]
#                 if ibc < idx_beg_coarse or ibc >= idx_beg_coarse + idx_num_coarse:
#                     continue

#                 phi_coarse = get_psi_coarse(ikc, ibc, isvalflag)
#                 overlap = np.multiply(phi_fine, np.conj(phi_coarse))

#                 cmtxel = np.sum(overlap) * self.gspace.reallat_dv
#                 cmtx[ibf - idx_beg_fine, ibc - idx_beg_coarse] = cmtxel
        
#         norms = np.sqrt(np.sum(np.abs(cmtx) ** 2, axis=-1, keepdims=True))
#         cmtx /= (norms + 1e-10)

#         return cmtx


from functools import lru_cache
from typing import List, NamedTuple
import numpy as np

from qtm.constants import ELECTRONVOLT_HART
from qtm.crystal import Crystal
from qtm.klist import KList
from qtm.gspace.gspc import GSpace
from qtm.gspace.gkspc import GkSpace
from qtm.dft.kswfn import KSWfn

from qtm.mpi.comm import MPI4PY_INSTALLED
from kernel import KernelMtxEl
from qtm.interfaces.bgw.wfn2py import WfnData
from qtm.interfaces.bgw.h5_utils import *
from qtm.gw.core import (
    QPoints,
    sort_cryst_like_BGW,
    reorder_2d_matrix_sorted_gvecs,
)

if MPI4PY_INSTALLED:
    from mpi4py import MPI


class InterpMtxEl:
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

        self.val_num_max_fine = max(list_val_idx_fine[1]) + 1
        self.val_num_max_coarse = max(list_val_idx_coarse[1]) + 1

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
            if num_val_bands_fine > self.val_num_max_fine:
                raise ValueError(
                    f"num_val_bands_fine {num_val_bands_fine} exceeds max {self.val_num_max_fine}."
                )
            self.val_num_fine = num_val_bands_fine
        else:
            self.val_num_fine = self.val_num_max_fine

        if num_val_bands_coarse is not None:
            if num_val_bands_coarse > self.val_num_max_coarse:
                raise ValueError(
                    f"num_val_bands_coarse {num_val_bands_coarse} exceeds max {self.val_num_max_coarse}."
                )
            self.val_num_coarse = num_val_bands_coarse
        else:
            self.val_num_coarse = self.val_num_max_coarse

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


    
    @classmethod
    def from_BGW(
        cls,
        wfn_finedata: WfnData,
        wfn_coarsedata: WfnData,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        kernel: KernelMtxEl,
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