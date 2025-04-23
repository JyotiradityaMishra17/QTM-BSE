import numpy as np
from typing import List, NamedTuple
from copy import deepcopy

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

from mtxelv2 import ChargeMtxELv2


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
        l_wfn: List[KSWfn],
        l_gsp_wfn: List[GkSpace],
        qpts: QPoints,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        l_epsmats: List[np.ndarray],
        q0val: np.ndarray,
        parallel: bool = True,
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

        self.list_idx_G0 = []

        self.l_epsinv = []
        for i_q in range(self.qpts.numq):
            epsinv = self.l_epsmats[i_q]
            epsinv = np.conjugate(epsinv)
            
            sort_order = sort_cryst_like_BGW(self.l_gq[i_q].gk_cryst, self.l_gq[i_q].gk_norm2)
            sort_order_eps = sort_cryst_like_BGW(self.l_gq_epsinv[i_q].gk_cryst, self.l_gq_epsinv[i_q].gk_norm2)
            zero_index = np.argmin(np.linalg.norm(self.l_gq[i_q].g_cryst.T[sort_order], axis = 1))

            # After sorting W, M, etc, new location of G0.
            idx_G0 = sort_order[zero_index]
            self.list_idx_G0.append(idx_G0)
                        
            # As given in the sigma code, I need to use another sorting.
            sort_order_QTM = np.argsort(sort_order_eps)

            # But my matrices, vcoul, etc. are sorted according to BGW.
            sort_order_BGW = sort_order_QTM[sort_order_eps]


            self.l_epsinv.append(
                reorder_2d_matrix_sorted_gvecs(epsinv, sort_order_BGW)
            )
            
        self.q0val = q0val
            
        self.charge_mtxel = ChargeMtxELv2(
            crystal,
            gspace,
            kpts,
            l_wfn,
            l_gsp_wfn,
            qpts,
            epsinp,
            sigmainp,
            num_bands_val,
            num_bands_con,
        )

        self.l_wcoul = []
        for i_q in range(self.qpts.numq):
            idx_G0 = self.list_idx_G0[i_q]
            numg = self.l_gq[i_q].size_g

            epsinv = self.l_epsinv[i_q]
            numg_eps = self.l_gq_epsinv[i_q].size_g

            q0norm2 = np.dot(np.conjugate(self.q0val), self.q0val)
            norm_array = self.l_gq[i_q].gk_norm2

            vqg = np.where(norm_array == 0, 1/q0norm2, 1/np.where(norm_array == 0, 1, norm_array))
            sort_order = sort_cryst_like_BGW(self.l_gq[i_q].gk_cryst, norm_array)
            vqg = vqg[sort_order]

            wcoul = np.zeros((numg, numg), dtype=complex)

            # Fill in the entire wcoul matrix.
            wcoul[:numg_eps, :numg_eps] = epsinv[:numg_eps, :numg_eps].T * vqg[:numg_eps]
            np.fill_diagonal(wcoul[numg_eps:numg, numg_eps:numg], vqg[numg_eps:numg])

            # Correct the wing matrix elements.
            if i_q == self.qpts.index_q0:
                wcoul[idx_G0, :numg_eps] = 0
                wcoul[:numg_eps, idx_G0] = 0
            else:
                qq = np.linalg.norm(self.qpts.cryst[i_q])
                wcoul[idx_G0, :numg_eps] = epsinv[:numg_eps, idx_G0] * vqg[:numg_eps] * qq
                wcoul[:numg_eps, idx_G0] = epsinv[idx_G0, :numg_eps] * vqg[idx_G0] * qq

            # Correct the head matrix elements.
            wcoul[idx_G0, idx_G0] = 1

            self.l_wcoul.append(wcoul)
            

    @classmethod
    def from_qtm(
        cls,
        crystal: Crystal,
        gspace: GSpace,
        kpts: KList,
        l_wfn_kgrp: List[List[KSWfn]],
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        epsilon: Epsilon,
        q0val: np.ndarray,
        parallel: bool = True,
        num_bands_val: int = None,
        num_bands_con: int = None,
    ):
        kernelclass = KernelMtxElv2(
            crystal=crystal,
            gspace=gspace,
            kpts=kpts,
            l_wfn=[wfn[0] for wfn in l_wfn_kgrp],
            l_gsp_wfn=[wfn[0].gkspc for wfn in l_wfn_kgrp],
            qpts=QPoints.from_cryst(kpts.recilat, epsinp.is_q0, *epsinp.qpts),
            epsinp=epsinp,
            sigmainp=sigmainp,
            l_epsmats=epsilon.l_epsinv,
            q0val=q0val,
            parallel=parallel,
            num_bands_val=num_bands_val,
            num_bands_con=num_bands_con,
        )

        return kernelclass
    
    @classmethod
    def from_BGW(
        cls,
        wfndata: WfnData,
        epsinp: NamedTuple,
        sigmainp: NamedTuple,
        l_epsmats: List[np.ndarray],
        q0val: np.ndarray,
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
            l_wfn = wfndata.l_wfn,
            l_gsp_wfn =  wfndata.l_gk,
            qpts = qpts, 
            epsinp = epsinp,
            sigmainp = sigmainp,
            l_epsmats = l_epsmats,
            q0val=q0val,
            parallel = parallel,
            num_bands_val = num_bands_val,
            num_bands_con = num_bands_con,
            )

        return kernelclass
    
    def kernel_mtxel(self, parallel: bool = True):
        numk = self.kpts.numk
        numq = self.qpts.numq

        list_k = self.kpts.cryst
        list_q = self.qpts.cryst

        is_q0 = self.qpts.is_q0
        q0norm2 = np.dot(np.conjugate(self.q0val), self.q0val)

        num_val = self.charge_mtxel.val_num
        num_con = self.charge_mtxel.con_num

        # Calculate the value of v(q + G) for q = 0.
        norm_array = self.l_gq[0].gk_norm2
        vq0g = np.where(norm_array == 0, 0, 1/np.where(norm_array == 0, 1, norm_array))

        sort_order0 = sort_cryst_like_BGW(self.l_gq[0].gk_cryst, norm_array)
        vq0g = vq0g[sort_order0]

        if self.in_parallel and parallel:

            proc_rank = self.comm.Get_rank()
            if proc_rank == 0:
                q_indices, kp_indices = np.meshgrid(np.arange(numq), np.arange(numk), indexing='ij')
                full_pairs = np.stack((q_indices.flatten(), kp_indices.flatten()), axis=1)

                chunks = np.array_split(full_pairs, self.comm.Get_size())
            else:
                chunks = None

            local_pairs = self.comm.scatter(chunks, root=0)
            local_results = []

            for iq, ikp in local_pairs:
                idx_G0 = self.list_idx_G0[iq]

                mvcp = self.charge_mtxel.mtxelv2(0, ikp, "mvc") # -> (v, c, g)
                mccp = self.charge_mtxel.mtxelv2(iq, ikp, "mccp") # -> (c, C, g)

                mvvp, ik = self.charge_mtxel.mtxelv2(iq, ikp, "mvvp", ret_k=True) # -> (v, V, g)
                mvc = self.charge_mtxel.mtxelv2(0, ik, "mvc") # -> (v, c, g)
                
                wcoul = self.l_wcoul[iq]
                numg_eps = self.l_gq_epsinv[iq].size_g

                # Compute the head matrix elements.
                einstrh = "vV, cC -> cCvV"
                head = np.einsum(einstrh, np.conj(mvvp[:, :, idx_G0]), mccp[:, :, idx_G0], optimize=True)

                # Compute the wing matrix elements.
                einstrw1 = "vVG, G, cC -> cCvV"
                mvvp_w1 = deepcopy(mvvp)

                mvvp_w1[:, :, idx_G0] = 0
                wing = np.einsum(einstrw1, np.conj(mvvp_w1[:, :, :numg_eps]), wcoul[idx_G0, :numg_eps], mccp[:, :, idx_G0], optimize=True)

                einstrw2 = "vV, g, cCg -> cCvV"
                mccp_w2 = deepcopy(mccp)

                mccp_w2[:, :, idx_G0] = 0
                wing += np.einsum(einstrw2, np.conj(mvvp[:, :, idx_G0]), wcoul[:numg_eps, idx_G0], mccp_w2[:, :, :numg_eps], optimize=True)

                # Compute the body matrix elements.
                einstrb = "vVG, gG, cCg -> cCvV"
                mvvp_b = deepcopy(mvvp)
                mvvp_b[:, :, idx_G0] = 0

                mccp_b = deepcopy(mccp)
                mccp_b[:, :, idx_G0] = 0
                body = np.einsum(einstrb, np.conj(mvvp_b), wcoul, mccp_b, optimize=True)

                # Compute the exchange matrix elements.
                einstrx = "vcg, g, VCg -> cCvV"
                exc = np.einsum(einstrx, np.conj(mvc), vq0g, mvcp, optimize=True)

                local_results.append((ik, ikp, head, wing, body, exc))

            if proc_rank != 0:
                self.comm.send(local_results, dest=0, tag=77)
                
                exc_result = self.comm.bcast(None, root=0)
                head_result = self.comm.bcast(None, root=0)

                wing_result = self.comm.bcast(None, root=0)
                body_result = self.comm.bcast(None, root=0)
            
            else:
                exc_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)
                head_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)

                wing_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)
                body_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)

                for result in local_results:
                    ik, ikp, head, wing, body, exc = result

                    head_mtx[ik, ikp] = head
                    wing_mtx[ik, ikp] = wing
                    body_mtx[ik, ikp] = body
                    exc_mtx[ik, ikp] = exc

                for source in range(1, self.comm_size):
                    remote_results = self.comm.recv(source=source, tag=77)

                    for result in remote_results:
                        ik, ikp, head, wing, body, exc = result

                        head_mtx[ik, ikp] = head
                        wing_mtx[ik, ikp] = wing
                        body_mtx[ik, ikp] = body
                        exc_mtx[ik, ikp] = exc
                
                self.comm.bcast(head_mtx, root=0)
                self.comm.bcast(wing_mtx, root=0)
                self.comm.bcast(body_mtx, root=0)
                self.comm.bcast(exc_mtx, root=0)

            return {"exc": exc_mtx, "head": head_mtx, "wings": wing_mtx, "body": body_mtx}
        
        else:
            exc_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)
            head_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)

            wing_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)
            body_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)

            for iq in range(numq):
                for ikp in range(numk):
                    idx_G0 = self.list_idx_G0[iq]
                    mvcp = self.charge_mtxel.mtxelv2(0, ikp, "mvc")
                    mccp = self.charge_mtxel.mtxelv2(iq, ikp, "mccp")

                    mvvp, ik = self.charge_mtxel.mtxelv2(iq, ikp, "mvvp", ret_k=True)
                    mvc = self.charge_mtxel.mtxelv2(0, ik, "mvc")

                    wcoul = self.l_wcoul[iq]
                    numg_eps = self.l_gq_epsinv[iq].size_g

                    # Compute the head matrix elements.
                    einstrh = "vV, cC -> cCvV"
                    head = np.einsum(einstrh, np.conj(mvvp[:, :, idx_G0]), mccp[:, :, idx_G0], optimize=True)
                    head_mtx[ik, ikp] = head

                    # Compute the wing matrix elements.
                    einstrw1 = "vVG, G, cC -> cCvV"
                    mvvp_w1 = deepcopy(mvvp)
                    mvvp_w1[:, :, idx_G0] = 0
                    wing = np.einsum(einstrw1, np.conj(mvvp_w1[:, :, :numg_eps]), wcoul[idx_G0, :numg_eps], mccp[:, :, idx_G0], optimize=True)

                    einstrw2 = "vV, g, cCg -> cCvV"
                    mccp_w2 = deepcopy(mccp)
                    mccp_w2[:, :, idx_G0] = 0
                    wing += np.einsum(einstrw2, np.conj(mvvp[:, :, idx_G0]), wcoul[:numg_eps, idx_G0], mccp_w2[:, :, :numg_eps], optimize=True)
                    wing_mtx[ik, ikp] = wing

                    # Compute the body matrix elements.
                    einstrb = "vVG, gG, cCg -> cCvV"
                    mvvp_b = deepcopy(mvvp)
                    mvvp_b[:, :, idx_G0] = 0

                    mccp_b = deepcopy(mccp)
                    mccp_b[:, :, idx_G0] = 0
                    body = np.einsum(einstrb, np.conj(mvvp_b), wcoul, mccp_b, optimize=True)
                    body_mtx[ik, ikp] = body

                    # Compute the exchange matrix elements.
                    einstrx = "vcg, g, VCg -> cCvV"
                    exc = np.einsum(einstrx, np.conj(mvc), vq0g, mvcp, optimize=True)
                    exc_mtx[ik, ikp] = exc
                    
            return {"exc": exc_mtx, "head": head_mtx, "wings": wing_mtx, "body": body_mtx}

            








    






        










