import numpy as np
from typing import List, NamedTuple

from qtm.constants import ELECTRONVOLT_RYD, RYDBERG_HART
from qtm.crystal import Crystal
from qtm.klist import KList
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import GSpace
from qtm.dft.kswfn import KSWfn
from qtm.gw.core import (
    QPoints,
    sort_cryst_like_BGW,
    reorder_2d_matrix_sorted_gvecs,
)
from qtm.gw.vcoul import Vcoul
from qtm.gw.epsilon import Epsilon
from qtm.mpi.comm import MPI4PY_INSTALLED
from qtm.interfaces.bgw.wfn2py import WfnData
from qtm.interfaces.bgw.h5_utils import *

if MPI4PY_INSTALLED:
    from mpi4py import MPI

from mtxelv2 import MtxElv2


class KernelMtxElv2:
    ryd = 1 / ELECTRONVOLT_RYD
    fixwings = True
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
        vcoul: Vcoul,
        l_epsmats: List[np.ndarray],
        parallel: bool = True,
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
        self.l_epsmats = l_epsmats
        self.num_bands_val = num_bands_val
        self.num_bands_con = num_bands_con

        self.in_parallel = False
        self.comm = None
        self.comm_size = None
        if parallel and MPI4PY_INSTALLED:
            self.comm = MPI.COMM_WORLD
            self.comm_size = self.comm.Get_size()
            if self.comm_size > 1:
                self.in_parallel = True

        self.l_gq: List[GkSpace] = []
        for i_q in range(qpts.numq):
            self.l_gq.append(
                GkSpace(
                    gwfn=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                    ecutwfn=self.epsinp.epsilon_cutoff * RYDBERG_HART,
                )
            )

        # Store the index of G = 0 for each q-point.
        idxG0 = np.zeros(self.qpts.numq, dtype=int)
        for q_idx in range(self.qpts.numq):
            idxG0[q_idx] = np.argmin(
                np.linalg.norm(
                    self.l_gq[q_idx].g_cryst.T[
                        sort_cryst_like_BGW(
                            self.l_gq[q_idx].gk_cryst,
                            self.l_gq[q_idx].gk_norm2,
                        )
                    ],
                    axis=1,
                )
            )
        self.idxG0 = idxG0

        self.vcoul = vcoul

        self.l_epsinv = []
        for i_q in range(self.qpts.numq):
            epsinv = self.l_epsmats[i_q]
            if self.fixwings:
                epsinv = self.vcoul.calculate_fixedeps(
                    epsinv, i_q, random_sample=False
                )
            sort_order = sort_cryst_like_BGW(
                self.l_gq[i_q].gk_cryst, self.l_gq[i_q].gk_norm2
            )
            self.l_epsinv.append(
                reorder_2d_matrix_sorted_gvecs(epsinv, np.argsort(sort_order))
            )

        self.mtxel = MtxElv2(
            crystal,
            gspace,
            kpts,
            kptsq,
            l_wfn,
            l_wfnq,
            l_gsp_wfn,
            l_gsp_wfnq,
            qpts,
            epsinp,
            sigmainp,
            num_bands_val,
            num_bands_con,
        )

        occ = []
        for k_idx in range(self.kpts.numk):
            occ.append(self.l_wfn[k_idx].occ)
        occ = np.array(occ)
        self.occ = occ[:, : self.sigmainp.number_bands]

        self.list_val_idx = np.where(self.occ == 1)
        self.list_con_idx = np.where(self.occ == 0)

        self.qval0 = np.linalg.norm((self.kptsq.cryst - self.kpts.cryst)[0])
        self.kernel_factor = KernelMtxElv2.ryd / ((crystal.reallat.cellvol) * self.qpts.numq)

    @classmethod
    def from_qtm(
        cls,
        crystal: Crystal,
        gspace: GSpace,
        kpts: KList,
        kptsq: KList,
        l_wfn_kgrp: List[List[KSWfn]],
        l_wfn_kgrp_q : List[List[KSWfn]],
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        vcoul: Vcoul,
        epsilon: Epsilon,
        parallel: bool = True,
        num_bands_val: int = None,
        num_bands_con: int = None,
    ):
        kernelclass = KernelMtxElv2(
            crystal=crystal,
            gspace=gspace,
            kpts=kpts,
            kptsq=kptsq,
            l_wfn=[wfn[0] for wfn in l_wfn_kgrp],
            l_wfnq=[wfn[0] for wfn in l_wfn_kgrp_q],
            l_gsp_wfn=[wfn[0].gkspc for wfn in l_wfn_kgrp],
            l_gsp_wfnq=[wfn[0].gkspc for wfn in l_wfn_kgrp_q],
            qpts=QPoints.from_cryst(kpts.recilat, epsinp.is_q0, *epsinp.qpts),
            epsinp=epsinp,
            sigmainp=sigmainp,
            vcoul=vcoul,
            l_epsmats=epsilon.l_epsinv,
            parallel=parallel,
            num_bands_val=num_bands_val,
            num_bands_con=num_bands_con,
        )

        return kernelclass
    
    @classmethod
    def from_BGW(
        cls,
        wfndata: WfnData,
        wfnqdata: WfnData,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        vcoul: Vcoul,
        l_epsmats: List[np.ndarray],
        parallel: bool = True,
        num_bands_val: int = None,
        num_bands_con: int = None,
    ):
        l_qpts = np.array(epsinp.qpts)
        l_qpts[0] *= 0
        qpts = QPoints.from_cryst(wfndata.kpts.recilat, None, *l_qpts)

        kernelclass = KernelMtxElv2(
            crystal = wfndata.crystal,
            gspace = wfndata.grho,
            kpts = wfndata.kpts,
            kptsq = wfnqdata.kpts,
            l_wfn = wfndata.l_wfn,
            l_wfnq = wfnqdata.l_wfn,
            l_gsp_wfn =  wfndata.l_gk,
            l_gsp_wfnq = wfnqdata.l_gk,
            qpts = qpts, # QPoints.from_cryst(wfndata.kpts.recilat, epsinp.is_q0, *epsinp.qpts),
            epsinp = epsinp,
            sigmainp = sigmainp,
            vcoul = vcoul,
            l_epsmats = l_epsmats,
            parallel = parallel,
            num_bands_val = num_bands_val,
            num_bands_con = num_bands_con,
            )

        return kernelclass


    def exc_block(self, idx_G0, vqg, M, Mp):

        vqg_n0 = np.delete(vqg, idx_G0)
        M_n0 = np.delete(M, idx_G0, axis=-1)
        M_conj_n0 = np.conj(M_n0)
        Mp_n0 = np.delete(Mp, idx_G0, axis=-1)

        einstr = "vcg, g, VCg -> vVcC"
        block = np.einsum(einstr, M_conj_n0, vqg_n0, Mp_n0, optimize=True)

        block *= self.kernel_factor

        return block

    def head_block(self, idx_G0, vqg, epsinv, MC, MV):
        vqg_0 = vqg[idx_G0]
        MC_0 = MC[..., idx_G0]
        MV_0 = MV[..., idx_G0]
        MC_conj_0 = np.conj(MC_0)
        epsinv_0_0 = epsinv[idx_G0, idx_G0]

        einstr = "Cc, Vv -> vVcC"
        block = np.einsum(
            einstr,
            MC_conj_0,
            MV_0,
            optimize=True,
        )
        block *= self.kernel_factor * epsinv_0_0 * vqg_0
        return block


    def wings_block(self, idx_G0, vqg, epsinv, MC, MV):
        vqg_0 = vqg[idx_G0]
        vqg_n0 = np.delete(vqg, idx_G0)

        MC_0 = MC[..., idx_G0]
        MC_n0 = np.delete(MC, idx_G0, axis=-1)
        MV_0 = MV[..., idx_G0]
        MV_n0 = np.delete(MV, idx_G0, axis=-1)
        MC_conj_0 = np.conj(MC_0)
        MC_conj_n0 = np.conj(MC_n0)

        epsinv_0_n0 = np.delete(epsinv[idx_G0, :], idx_G0)
        epsinv_n0_0 = np.delete(epsinv[:, idx_G0], idx_G0)

        einstr_w1 = "Ccg, g, Vv -> vVcC"
        block = np.einsum(
            einstr_w1,
            MC_conj_n0,
            epsinv_n0_0,
            MV_0,
            optimize=True,
        )
        block *= vqg_0

        einstr_w2 = "Cc, G, G, VvG -> vVcC"
        block += np.einsum(
            einstr_w2,
            MC_conj_0,
            epsinv_0_n0,
            vqg_n0,
            MV_n0,
            optimize=True,
        )

        block *= self.kernel_factor

        return block
    

    def body_block(self, idx_G0, vqg, epsinv, MC, MV):
        vqg_n0 = np.delete(vqg, idx_G0)
        MC_n0 = np.delete(MC, idx_G0, axis=-1)
        MV_n0 = np.delete(MV, idx_G0, axis=-1)
        MC_conj_n0 = np.conj(MC_n0)

        epsinv_all_n0 = np.delete(epsinv, idx_G0, axis=-1)
        epsinv_n0_n0 = np.delete(epsinv_all_n0, idx_G0, axis=0)

        einstr = "Ccg, gG, G, VvG ->vVcC"
        block = np.einsum(
            einstr,
            MC_conj_n0,
            epsinv_n0_n0,
            vqg_n0,
            MV_n0,
            optimize=True,
        )
        block *= self.kernel_factor

        return block

    def dir_block(self, vqg, epsinv, MC, MV):
        MC_conj = np.conj(MC)
        einstr = "Ccg, gG, G, VvG -> vVcC"
        block = np.einsum(
            einstr,
            MC_conj,
            epsinv,
            vqg,
            MV,
            optimize=True,
        )
        block *= self.kernel_factor

        return block

    def kernel_mtxel(self, parallel: bool = True):
        numk = self.kpts.numk
        numq = self.qpts.numq
        vq0g = self.vcoul.vcoul[0]

        list_k = self.kpts.cryst
        list_q = self.qpts.cryst
        is_q0 = self.qpts.is_q0

        num_val = self.mtxel.val_num
        num_con = self.mtxel.con_num

        if self.in_parallel and parallel:
            proc_rank = self.comm.Get_rank()

            if proc_rank == 0:
                q_indices, kp_indices = np.meshgrid(
                    np.arange(numq),
                    np.arange(numk),
                    indexing="ij",
                )
                full_pairs = np.stack(
                    (q_indices.flatten(), kp_indices.flatten()), axis=1
                )
                chunks = np.array_split(full_pairs, self.comm.Get_size())
            else:
                chunks = None

            local_pairs = self.comm.scatter(chunks, root=0)
            local_blocks = []  # Each element: (k_idx, kp_idx, exc_block, dir_block)

            for pair in local_pairs:
                q_idx, kp_idx = pair
                qpt = list_q[q_idx]
                is_qpt_0 = None if is_q0 == None else is_q0[q_idx]

                idx_G0 = self.idxG0[q_idx]
                vqg = self.vcoul.vcoul[q_idx]
                epsinv = self.l_epsinv[q_idx]

                if is_qpt_0:
                    umklapp = -np.floor(np.around(list_k, 5))
                    list_kpq = list_k + umklapp
                else:
                    umklapp = -np.floor(np.around(list_k + qpt, 5))
                    list_kpq = list_k + qpt + umklapp

                kpq = list_kpq[kp_idx]
                list_idx_match = np.nonzero(
                    np.all(
                        np.isclose(
                            list_k, kpq[None, :],
                            atol=KernelMtxElv2.TOLERANCE,
                        ),
                        axis=1,
                    )
                )[0]

                for k_idx in list_idx_match:
                    M = self.mtxel.mtxel_s2(
                        0, k_idx, bra="val", ket="con"
                    )
                    Mp = self.mtxel.mtxel_s2(
                        0, kp_idx, bra="val", ket="con"
                    )
                    MC = self.mtxel.mtxel_s2(
                        q_idx, k_idx, bra="con", ket="con"
                    )
                    MV = self.mtxel.mtxel_s2(
                        q_idx, k_idx, bra="val", ket="val"
                    )

                    exc_block = self.exc_block(idx_G0, vq0g, M, Mp) 
                    dir_block = self.dir_block(vqg, epsinv, MC, MV)
                    local_blocks.append((k_idx, kp_idx, exc_block, dir_block))

            if proc_rank != 0:
                self.comm.send(local_blocks, dest=0, tag=77)
                exc_result = self.comm.bcast(None, root=0)
                dir_result = self.comm.bcast(None, root=0)
                
            else:
                exc_mtx = np.zeros((numk, numk, num_val, num_val, 
                                num_con, num_con), dtype=complex)
                dir_mtx = np.zeros((numk, numk, num_val, num_val,
                                num_con, num_con), dtype=complex)

                for k_idx, kp_idx, exc_block, dir_block in local_blocks:
                    exc_mtx[k_idx, kp_idx] += exc_block
                    dir_mtx[k_idx, kp_idx] += dir_block

                for source in range(1, self.comm.Get_size()):
                    remote_blocks = self.comm.recv(source=source, tag=77)
                    for k_idx, kp_idx, exc_block, dir_block in remote_blocks:
                        exc_mtx[k_idx, kp_idx] += exc_block
                        dir_mtx[k_idx, kp_idx] += dir_block

                exc_result = exc_mtx
                exc_result = self.comm.bcast(exc_result, root=0)

                dir_result = dir_mtx
                dir_result = self.comm.bcast(dir_result, root=0)

            return exc_result, dir_result

        else:
            exc_mtx = np.zeros((numk, numk, num_val, num_val, 
                            num_con, num_con), dtype=complex)
            dir_mtx = np.zeros((numk, numk, num_val, num_val,
                            num_con, num_con), dtype=complex)

            for q_idx in range(numq):
                idx_G0 = self.idxG0[q_idx]
                vqg = self.vcoul.vcoul[q_idx]
                epsinv = self.l_epsinv[q_idx]

                qpt = list_q[q_idx]
                is_qpt_0 = None if is_q0 == None else is_q0[q_idx]

                for kp_idx in range(numk):
                    if is_qpt_0:
                        umklapp = -np.floor(np.around(list_k, 5))
                        list_kpq = list_k + umklapp
                    else:
                        umklapp = -np.floor(np.around(list_k + qpt, 5))
                        list_kpq = list_k + qpt + umklapp

                    kpq = list_kpq[kp_idx]
                    list_idx_match = np.nonzero(
                        np.all(
                            np.isclose(
                                list_k, kpq[None, :],
                                atol=KernelMtxElv2.TOLERANCE,
                            ),
                            axis=1,
                        )
                    )[0]

                    for k_idx in list_idx_match:
                        M = self.mtxel.mtxel_s2(
                            0, k_idx, bra="val", ket="con"
                        )
                        Mp = self.mtxel.mtxel_s2(
                            0, kp_idx, bra="val", ket="con"
                        )
                        MC = self.mtxel.mtxel_s2(
                            q_idx, k_idx, bra="con", ket="con"
                        )
                        MV = self.mtxel.mtxel_s2(
                            q_idx, k_idx, bra="val", ket="val"
                        )

                        exc_block = self.exc_block(idx_G0, vq0g, M, Mp) 
                        dir_block = self.dir_block(vqg, epsinv, MC, MV)

                        exc_mtx[k_idx, kp_idx] += exc_block
                        dir_mtx[k_idx, kp_idx] += dir_block

            return exc_mtx, dir_mtx
