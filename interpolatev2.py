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
from kernelv2 import KernelMtxElv2
from qtm.interfaces.bgw.wfn2py import WfnData
from qtm.interfaces.bgw.h5_utils import *
from qtm.gw.core import (
    QPoints,
    sort_cryst_like_BGW,
    reorder_2d_matrix_sorted_gvecs,
)

if MPI4PY_INSTALLED:
    from mpi4py import MPI


class InterpMtxElv2:
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
        kernel: KernelMtxElv2,
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

        self.list_val_idx_fine = np.where(self.occ_fine == 1)
        self.list_val_idx_coarse = np.where(self.occ_coarse == 1)
        self.list_con_idx_fine = np.where(self.occ_fine == 0)
        self.list_con_idx_coarse = np.where(self.occ_coarse == 0)

        self.num_bands = self.sigmainp.number_bands
        self.min_band = self.sigmainp.band_index_min
        self.max_band = self.sigmainp.band_index_max

        self.val_idx_beg_coarse = 0
        self.con_idx_beg_coarse = min(self.list_con_idx_coarse[1])
        val_num_coarse_max = max(self.list_val_idx_coarse[1]) + 1
        con_num_coarse_max = self.max_band - self.con_idx_beg_coarse

        if num_val_bands_coarse is not None:
            if num_val_bands_coarse > val_num_coarse_max:
                raise ValueError(
                    f"num_val_bands_coarse {num_val_bands_coarse} exceeds max {val_num_coarse_max}."
                )
            self.val_num_coarse = num_val_bands_coarse
        else:
            self.val_num_coarse = val_num_coarse_max

        if num_con_bands_coarse is not None:
            if num_con_bands_coarse > con_num_coarse_max:
                raise ValueError(
                    f"num_con_bands_coarse {num_con_bands_coarse} exceeds max {con_num_coarse_max}."
                )
            self.con_num_coarse = num_con_bands_coarse
        else:
            self.con_num_coarse = con_num_coarse_max

        self.val_idx_beg_fine = 0
        self.con_idx_beg_fine = min(self.list_con_idx_fine[1])
        val_num_fine_max = max(self.list_val_idx_fine[1]) + 1
        con_num_fine_max = self.max_band - self.con_idx_beg_fine

        if num_val_bands_fine is not None:
            if num_val_bands_fine > val_num_fine_max:
                raise ValueError(
                    f"num_val_bands_fine {num_val_bands_fine} exceeds max {val_num_fine_max}."
                )
            self.val_num_fine = num_val_bands_fine
        else:
            self.val_num_fine = val_num_fine_max
        
        if num_con_bands_fine is not None:
            if num_con_bands_fine > con_num_fine_max:
                raise ValueError(
                    f"num_con_bands_fine {num_con_bands_fine} exceeds max {con_num_fine_max}."
                )
            self.con_num_fine = num_con_bands_fine
        else:
            self.con_num_fine = con_num_fine_max

    @classmethod
    def from_qtm(
        cls,
        crystal: Crystal,
        gspace: GSpace,
        fine_kpts: KList,
        coarse_kpts: KList,
        fine_l_wfn_kgrp: List[List[KSWfn]],
        coarse_l_wfn_kgrp: List[List[KSWfn]],
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        kernel: KernelMtxElv2,
        parallel: bool = True,
        num_val_bands_fine: int = None,
        num_con_bands_fine: int = None,
        num_val_bands_coarse: int = None,
        num_con_bands_coarse: int = None,
    ):
        interpclass = InterpMtxElv2(
            crystal=crystal,
            gspace=gspace,
            fine_kpts=fine_kpts,
            coarse_kpts=coarse_kpts,
            fine_l_wfn=[wfn[0] for wfn in fine_l_wfn_kgrp],
            coarse_l_wfn=[wfn[0] for wfn in coarse_l_wfn_kgrp],
            fine_l_gsp_wfn=[wfn[0].gkspc for wfn in fine_l_wfn_kgrp],
            coarse_l_gsp_wfn=[wfn[0].gkspc for wfn in coarse_l_wfn_kgrp],
            epsinp=epsinp,
            sigmainp=sigmainp,
            kernel=kernel,
            parallel=parallel,
            num_val_bands_fine=num_val_bands_fine,
            num_con_bands_fine=num_con_bands_fine,
            num_val_bands_coarse=num_val_bands_coarse,
            num_con_bands_coarse=num_con_bands_coarse,
        )

        return interpclass
    
    @classmethod
    def from_BGW(
        cls,
        wfn_finedata: WfnData,
        wfn_coarsedata: WfnData,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        kernel: KernelMtxElv2,
        parallel: bool = True,
        num_val_bands_fine: int = None,
        num_con_bands_fine: int = None,
        num_val_bands_coarse: int = None,
        num_con_bands_coarse: int = None,
    ):
        interplcass = InterpMtxElv2(
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
            wfn_fine = self.fine_l_wfn[k_idx_fine]
            phi_fine = np.zeros(wfn_fine.gkspc.grid_shape, dtype=complex)
            self.fine_l_gsp_wfn[k_idx_fine]._fft.g2r(
                arr_inp=wfn_fine.evc_gk.data[band_idx_fine, :],
                arr_out=phi_fine,
            )
            return phi_fine

        @lru_cache(maxsize=int(idx_num_coarse))
        def get_eigvec_coarse_gk_r2g(k_idx_coarse, band_idx_coarse):
            wfn_coarse = self.coarse_l_wfn[k_idx_coarse]
            phi_coarse = np.zeros(wfn_coarse.gkspc.grid_shape, dtype=complex)
            self.coarse_l_gsp_wfn[k_idx_coarse]._fft.g2r(
                arr_inp=wfn_coarse.evc_gk.data[band_idx_coarse, :],
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

    def eqp_fine_mtxel(self, eqp_crse: np.ndarray,
                  type: str, parallel: bool = True):
        if type not in ["val", "con"]:
            raise ValueError("type must be either 'val' or 'con'")

        list_k_fine = self.fine_kpts.cryst
        list_k_coarse = self.coarse_kpts.cryst
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

        
        # Quasiparticle energies.
        eqp_coarse = eqp_crse[:, idx_beg_coarse : idx_beg_coarse + idx_num_coarse]

        # Mean field energies.
        emf_factor = 1 / ELECTRONVOLT_HART
        emf_fine = np.array(
            [self.fine_l_wfn[k_idx].evl for k_idx in range(numk_fine)]
        )
        emf_fine = emf_fine[:, idx_beg_fine : idx_beg_fine + idx_num_fine] * emf_factor

        emf_coarse = np.array(
            [self.coarse_l_wfn[k_idx].evl for k_idx in range(numk_coarse)]
        )
        emf_coarse = emf_coarse[:, idx_beg_coarse : idx_beg_coarse + idx_num_coarse] * emf_factor

        eqp_mtx = np.zeros((numk_fine, idx_num_fine))

        def get_block_idx(arr, n=4):
            blocks = []
            i = 0
            while i < len(arr):
                start = i
                val = arr[i]
                while i < len(arr) and arr[i] == val:
                    i += 1
                end = i
                blocks.append((start, end, val))
            blocks_sorted = sorted(blocks, key=lambda x: (x[2], x[0]))
            selected_blocks = blocks_sorted[:n]
            result = [list(range(start, end)) for start, end,
                       _ in selected_blocks]
            return result
        
        def barycentric_weights(v1, v2, v3, v4, p):
            # Determine the tetrahedral weights
            T = np.column_stack((v2 - v1, v3 - v1, v4 - v1))
            weights = np.dot(np.linalg.pinv(T), (p - v1))

            w2, w3, w4 = weights
            w1 = 1 - (w2 + w3 + w4)
            
            return np.array([w1, w2, w3, w4])

        def eqp_mtxel_kf(idx_fine):
            k_idx_fine = list_idx_fine[0][idx_fine]
            band_idx_fine = list_idx_fine[1][idx_fine]

            if band_idx_fine < idx_beg_fine or band_idx_fine >= idx_beg_fine + idx_num_fine:
                return None


            list_kpt_coarse = list_k_coarse[list_idx_coarse[0][:], :]
            kpt_fine = list_k_fine[k_idx_fine]

            distances = np.linalg.norm(
                list_kpt_coarse - kpt_fine[None, :], axis=1
            )
            
            selected_idx = get_block_idx(distances, n = 4)
            selected_k = []
            for nblk in range(4):
                idx = selected_idx[nblk][0]
                k_idx_coarse_selected = list_idx_coarse[0][idx]

                kpt_coarse_selected = list_k_coarse[k_idx_coarse_selected]
                kpt_coarse_selected = np.array(kpt_coarse_selected)

                selected_k.append(kpt_coarse_selected)

            kpt_fine = np.array(kpt_fine)
            weights = barycentric_weights(
                selected_k[0], selected_k[1], 
                selected_k[2], selected_k[3], kpt_fine
            )

            avg_corr = 0.0
            for nblk in range(4):
                for idx_coarse in selected_idx[nblk]:
                    k_idx_coarse = list_idx_coarse[0][idx_coarse]
                    band_idx_coarse = list_idx_coarse[1][idx_coarse]

                    if band_idx_coarse < idx_beg_coarse or band_idx_coarse >= idx_beg_coarse + idx_num_coarse:
                        continue

                    coeffmat = self.coeff_mtxel(
                        k_idx_fine, k_idx_coarse, type)
                    
                    coeff = coeffmat[band_idx_fine - idx_beg_fine, band_idx_coarse - idx_beg_coarse]

                    coeff2 = np.abs(coeff) ** 2
                    avg_corr += weights[nblk] * coeff2 * (
                        eqp_coarse[k_idx_coarse, band_idx_coarse - idx_beg_coarse] -
                        emf_coarse[k_idx_coarse, band_idx_coarse - idx_beg_coarse]
                    )                    

            computed_eqp = emf_fine[k_idx_fine, 
                                    band_idx_fine - idx_beg_fine] + avg_corr
            return (k_idx_fine, band_idx_fine - idx_beg_fine, computed_eqp)
        

        # --- Serial Version ---
        if not (self.in_parallel and parallel):
            contributions = []
            for idx_fine in range(len(list_idx_fine[0])):
                contrib = eqp_mtxel_kf(idx_fine)
                if contrib is not None:
                    contributions.append(contrib)
            for (kf, bf, val) in contributions:
                eqp_mtx[kf, bf] = val

        # --- Parallel Version ---
        else:
            proc_rank = self.comm.Get_rank()
            proc_size = self.comm.Get_size()
            if proc_rank == 0:
                indices = np.arange(len(list_idx_fine[0]))
                chunks = np.array_split(indices, proc_size)
            else:
                chunks = None
            local_indices = self.comm.scatter(chunks, root=0)

            local_contribs = []
            for idx_fine in local_indices:
                contrib = eqp_mtxel_kf(idx_fine)
                if contrib is not None:
                    local_contribs.append(contrib)

            if proc_rank != 0:
                self.comm.send(local_contribs, dest=0, tag=77)
                eqp_mtx = self.comm.bcast(None, root=0)
            else:
                all_contribs = local_contribs
                for source in range(1, proc_size):
                    remote_contribs = self.comm.recv(source=source, tag=77)
                    all_contribs.extend(remote_contribs)
                for (kf, bf, val) in all_contribs:
                    eqp_mtx[kf, bf] = val
                eqp_mtx = self.comm.bcast(eqp_mtx, root=0)

        return eqp_mtx   
    
    def closest_kpt_coeffs(self, parallel: bool = True):
        numk_fine = self.fine_kpts.numk
        list_k_fine = self.fine_kpts.cryst
        list_k_coarse = self.coarse_kpts.cryst

        num_val_fine = self.val_num_fine
        num_con_fine = self.con_num_fine
        num_val_coarse = self.val_num_coarse
        num_con_coarse = self.con_num_coarse

        closest_array = np.zeros(numk_fine, dtype=int)
        closest_coeff_val = np.zeros((numk_fine, num_val_fine, num_val_coarse), dtype=complex)
        closest_coeff_con = np.zeros((numk_fine, num_con_fine, num_con_coarse), dtype=complex)

        for k_idx_fine in range(numk_fine):
            kpt_fine = list_k_fine[k_idx_fine]

            distances = np.linalg.norm(
                list_k_coarse - kpt_fine[None, :], axis=1
            )

            idx_closest = np.argmin(distances)
            closest_array[k_idx_fine] = idx_closest  
            closest_coeff_val[k_idx_fine, :, :] = self.coeff_mtxel(
                k_idx_fine, idx_closest, type="val"
            )
            closest_coeff_con[k_idx_fine, :, :] = self.coeff_mtxel(
                k_idx_fine, idx_closest, type="con"
            )
        
        return closest_array, closest_coeff_val, closest_coeff_con
    
    def kernel_fine_mtxel(
            self,
            closest_array: np.ndarray,
            closest_coeff_val: np.ndarray,
            closest_coeff_con: np.ndarray,
            parallel: bool = True,
    ):
        numk_fine = self.fine_kpts.numk
        numk_coarse = self.coarse_kpts.numk
        numq = self.kernel.qpts.numq

        list_k_coarse = self.coarse_kpts.cryst
        list_q = self.kernel.qpts.cryst
        is_q0 = self.kernel.qpts.is_q0
        vq0g = self.kernel.vcoul.vcoul[0]

        num_val_fine = self.val_num_fine
        num_con_fine = self.con_num_fine
      
        if self.in_parallel and parallel:
            proc_rank = self.comm.Get_rank()

            if proc_rank == 0:
                q_indices, coarse_kp_indices = np.meshgrid(
                    np.arange(numq), 
                    np.arange(numk_coarse), 
                    indexing='ij'
                )
                full_pairs = np.stack(
                    (q_indices.flatten(), coarse_kp_indices.flatten()), axis=1
                )
                chunks = np.array_split(full_pairs, self.comm.Get_size())
            else:
                chunks = None

            local_pairs = self.comm.scatter(chunks, root=0)
            local_blocks = []

            for pair in local_pairs:
                q_idx, coarse_kp_idx = pair
                qpt = list_q[q_idx]
                qpt_val = np.linalg.norm(qpt)

                is_qpt_0 = None if is_q0 == None else is_q0[q_idx]
                list_fine_kp_idx = np.where(closest_array == coarse_kp_idx)[0]

                idx_G0 = self.kernel.idxG0[q_idx]
                vqg = self.kernel.vcoul.vcoul[q_idx]
                epsinv = self.kernel.l_epsinv[q_idx]

                if is_qpt_0:
                    umklapp = -np.floor(np.around(list_k_coarse, 5))
                    coarse_list_kpq = list_k_coarse + umklapp
                else:
                    umklapp = -np.floor(np.around(list_k_coarse + qpt, 5))
                    coarse_list_kpq = list_k_coarse + qpt + umklapp

                coarse_kpq = coarse_list_kpq[coarse_kp_idx]
                coarse_list_idx_match = np.nonzero(
                    np.all(
                        np.isclose(
                            list_k_coarse, coarse_kpq[None, :],
                            atol=KernelMtxElv2.TOLERANCE,
                        ),
                        axis=1,
                    )
                )[0]

                for coarse_k_idx in coarse_list_idx_match:
                    M = self.kernel.mtxel.mtxel_s2(0, coarse_k_idx,
                                                   bra = "val", ket = "con")
                    Mp = self.kernel.mtxel.mtxel_s2(0, coarse_kp_idx,
                                                    bra = "val", ket = "con")
                    
                    MC = self.kernel.mtxel.mtxel_s2(q_idx, coarse_k_idx,
                                                    bra = "con", ket = "con")
                    
                    MV = self.kernel.mtxel.mtxel_s2(q_idx, coarse_k_idx,
                                                   bra = "val", ket = "val")
                    
                    exc = self.kernel.exc_block(idx_G0, vq0g, M, Mp)
                    head = self.kernel.head_block(idx_G0, vqg, epsinv, MC, MV) * (qpt_val**2 + 1e-10)
                    wings = self.kernel.wings_block(idx_G0, vqg, epsinv, MC, MV) * (qpt_val + 1e-10)
                    body = self.kernel.body_block(idx_G0, vqg, epsinv, MC, MV)

                    list_fine_k_idx = np.where(closest_array == coarse_k_idx)[0]

                    for fine_k_idx in list_fine_k_idx:
                        coeff_k_val = closest_coeff_val[fine_k_idx]
                        coeff_k_val_conj = np.conj(coeff_k_val)

                        coeff_k_con = closest_coeff_con[fine_k_idx] 

                        for fine_kp_idx in list_fine_kp_idx:
                            coeff_kp_val = closest_coeff_val[fine_kp_idx]

                            coeff_kp_con = closest_coeff_con[fine_kp_idx]
                            coeff_kp_con_conj = np.conj(coeff_kp_con)

                            einstr = "cb, va, CB, VA, aAbB -> vVcC"

                            new_head = np.einsum(
                                einstr,
                                coeff_k_con,
                                coeff_k_val_conj,
                                coeff_kp_con_conj,
                                coeff_kp_val,
                                head,
                                optimize=True,
                            )

                            new_wings = np.einsum(
                                einstr,
                                coeff_k_con,
                                coeff_k_val_conj,
                                coeff_kp_con_conj,
                                coeff_kp_val,
                                wings,
                                optimize=True,
                            )

                            new_body = np.einsum(
                                einstr,
                                coeff_k_con,
                                coeff_k_val_conj,
                                coeff_kp_con_conj,
                                coeff_kp_val,
                                body,
                                optimize=True,
                            )

                            new_exc = np.einsum(
                                einstr,
                                coeff_k_con,
                                coeff_k_val_conj,
                                coeff_kp_con_conj,
                                coeff_kp_val,
                                exc,
                                optimize=True,
                            )

                            new_head = new_head / (qpt_val**2 + 1e-10)
                            new_wings = new_wings / (qpt_val + 1e-10)

                            dir_block = new_head + new_wings + new_body
                            exc_block = new_exc

                            local_blocks.append(
                                (fine_k_idx, fine_kp_idx, exc_block, dir_block)
                            )

            if proc_rank != 0:
                self.comm.send(local_blocks, dest=0, tag=77)
                exc_result = self.comm.bcast(None, root=0)
                dir_result = self.comm.bcast(None, root=0)

            else:
                exc_mtx = np.zeros(
                    (numk_fine, numk_fine, num_val_fine, num_val_fine,
                     num_con_fine, num_con_fine), dtype=complex
                )

                dir_mtx = np.zeros(
                    (numk_fine, numk_fine, num_val_fine, num_val_fine,
                     num_con_fine, num_con_fine), dtype=complex
                )

                for fine_k_idx, fine_kp_idx, exc_block, dir_block in local_blocks:
                    exc_mtx[fine_k_idx, fine_kp_idx] += exc_block
                    dir_mtx[fine_k_idx, fine_kp_idx] += dir_block

                for source in range(1, self.comm.Get_size()):
                    remote_blocks = self.comm.recv(source=source, tag=77)
                    for fine_k_idx, fine_kp_idx, exc_block, dir_block in remote_blocks:
                        exc_mtx[fine_k_idx, fine_kp_idx] += exc_block
                        dir_mtx[fine_k_idx, fine_kp_idx] += dir_block

                exc_result = self.comm.bcast(exc_mtx, root=0)
                dir_result = self.comm.bcast(dir_mtx, root=0)

            return exc_result, dir_result
        
        else:
            exc_mtx = np.zeros(
                (numk_fine, numk_fine, num_val_fine, num_val_fine,
                 num_con_fine, num_con_fine), dtype=complex
            )

            dir_mtx = np.zeros(
                (numk_fine, numk_fine, num_val_fine, num_val_fine,
                 num_con_fine, num_con_fine), dtype=complex
            )


            for q_idx in range(numq):
                for coarse_kp_idx in range(numk_coarse):
                    qpt = list_q[q_idx]
                    qpt_val = np.linalg.norm(qpt)

                    is_qpt_0 = None if is_q0 == None else is_q0[q_idx]
                    list_fine_kp_idx = np.where(closest_array == coarse_kp_idx)[0]

                    idx_G0 = self.kernel.idxG0[q_idx]
                    vqg = self.kernel.vcoul.vcoul[q_idx]
                    epsinv = self.kernel.l_epsinv[q_idx]

                    if is_qpt_0:
                        umklapp = -np.floor(np.around(list_k_coarse, 5))
                        coarse_list_kpq = list_k_coarse + umklapp
                    else:
                        umklapp = -np.floor(np.around(list_k_coarse + qpt, 5))
                        coarse_list_kpq = list_k_coarse + qpt + umklapp

                    coarse_kpq = coarse_list_kpq[coarse_kp_idx]
                    coarse_list_idx_match = np.nonzero(
                        np.all(
                            np.isclose(
                                list_k_coarse, coarse_kpq[None, :],
                                atol=KernelMtxElv2.TOLERANCE,
                            ),
                            axis=1,
                        )
                    )[0]

                    for coarse_k_idx in coarse_list_idx_match:
                        M = self.kernel.mtxel.mtxel_s2(0, coarse_k_idx,
                                                       bra="val", ket="con")
                        
                        Mp = self.kernel.mtxel.mtxel_s2(0, coarse_kp_idx,
                                                        bra="val", ket="con")
                        
                        MC = self.kernel.mtxel.mtxel_s2(q_idx, coarse_k_idx,
                                                        bra="con", ket="con")
                        
                        MV = self.kernel.mtxel.mtxel_s2(q_idx, coarse_k_idx,
                                                       bra="val", ket="val")
                        
                        exc = self.kernel.exc_block(idx_G0, vq0g, M, Mp)
                        head = self.kernel.head_block(idx_G0, vqg, epsinv, MC, MV) * (qpt_val**2 + 1e-10)
                        wings = self.kernel.wings_block(idx_G0, vqg, epsinv, MC, MV) * (qpt_val + 1e-10)
                        body = self.kernel.body_block(idx_G0, vqg, epsinv, MC, MV)

                        list_fine_k_idx = np.where(closest_array == coarse_k_idx)[0]

                        for fine_k_idx in list_fine_k_idx:
                            coeff_k_val = closest_coeff_val[fine_k_idx]
                            coeff_k_val_conj = np.conj(coeff_k_val)

                            coeff_k_con = closest_coeff_con[fine_k_idx] 

                            for fine_kp_idx in list_fine_kp_idx:
                                coeff_kp_val = closest_coeff_val[fine_kp_idx]

                                coeff_kp_con = closest_coeff_con[fine_kp_idx]
                                coeff_kp_con_conj = np.conj(coeff_kp_con)

                                einstr = "cb, va, CB, VA, aAbB -> vVcC"

                                new_head = np.einsum(
                                    einstr,
                                    coeff_k_con,
                                    coeff_k_val_conj,
                                    coeff_kp_con_conj,
                                    coeff_kp_val,
                                    head,
                                    optimize=True,
                                )

                                new_wings = np.einsum(
                                    einstr,
                                    coeff_k_con,
                                    coeff_k_val_conj,
                                    coeff_kp_con_conj,
                                    coeff_kp_val,
                                    wings,
                                    optimize=True,
                                )

                                new_body = np.einsum(
                                    einstr,
                                    coeff_k_con,
                                    coeff_k_val_conj,
                                    coeff_kp_con_conj,
                                    coeff_kp_val,
                                    body,
                                    optimize=True,
                                )

                                new_exc = np.einsum(
                                    einstr,
                                    coeff_k_con,
                                    coeff_k_val_conj,
                                    coeff_kp_con_conj,
                                    coeff_kp_val,
                                    exc,
                                    optimize=True,
                                )

                                new_head = new_head / (qpt_val**2 + 1e-10)
                                new_wings = new_wings / (qpt_val + 1e-10)

                                exc_block = new_exc
                                dir_block = new_head + new_wings + new_body

                                exc_mtx[fine_k_idx, fine_kp_idx] += exc_block
                                dir_mtx[fine_k_idx, fine_kp_idx] += dir_block

            return exc_mtx, dir_mtx
        
    def construct_HBSE(
            self,
            kernel_mtx: np.ndarray,
            eqp_mtx_val: np.ndarray,
            eqp_mtx_con: np.ndarray,
    ):
        numk_fine = self.fine_kpts.numk
        num_val_fine = self.val_num_fine
        num_con_fine = self.con_num_fine

        BSE_mtx = np.zeros(
            (numk_fine, numk_fine, num_val_fine, num_val_fine,
             num_con_fine, num_con_fine), dtype=complex
        )

        k_idx = np.arange(numk_fine)
        v_idx = np.arange(num_val_fine)
        c_idx = np.arange(num_con_fine)

        diagvalues = eqp_mtx_con[:, np.newaxis, :] - eqp_mtx_val[:, :, np.newaxis]
        BSE_mtx[
            k_idx[:, None, None],
            k_idx[:, None, None],
            v_idx[None, :, None],
            v_idx[None, :, None],
            c_idx[None, None, :],
            c_idx[None, None, :],
        ] = diagvalues

        BSE_mtx += kernel_mtx

        # Reshape it into a 2D Array.
        BSE_mtx_2D = BSE_mtx.transpose(0, 2, 4, 1, 3, 5)
        HBSE_mtx = BSE_mtx_2D.reshape(
            numk_fine * num_val_fine * num_con_fine,
            numk_fine * num_val_fine * num_con_fine,
        )

        return HBSE_mtx
    
    



                