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

from mtxel import ChargeMtxEL


class KernelMtxEl:
    '''
    Class to compute the kernel matrix elements for the Bethe-Salpeter equation (BSE).
    The class computes the head, body and wings of the direct kernel as well as the exchange kernel as follows:

        a. Head: < vck | K^{d, head} | vck' > = M*_{vv'}(k, k', G = 0) * W_{G = 0, G' = 0}(k - k', 0) * M_{cc'}(k, k', G' = 0)

        b. Body: < vck | K^{d, body} | vck' > = (Sum over G != 0, G' != 0) M*_{vv'}(k, k', G) * W_{G, G'}(k - k', 0) * M_{cc'}(k, k', G')

        c. Wings: < vck | K^{d, wings} | vck' > = (Sum over G != 0) M*_{vv'}(k, k', G) * W_{G, G' = 0}(k - k', 0) * M_{cc'}(k, k', G' = 0) + (Sum over G' != 0) M*_{vv'}(k, k', G = 0) * W_{G = 0, G'}(k - k', 0) * M_{cc'}(k, k', G')
        
        d. Exchange: < vck | K^{d, exc} | vck' > = (Sum over G != 0) M*_{vc}(k, k, G) * V(G) * M_{v'c'}(k', k', G)
    
    The different kernel elements are adjusted to treat the q -> 0 limit as per the prescription in Table 2 of "BerkeleyGW: A Massively ...".    
    
    '''
    ryd = 1 / ELECTRONVOLT_RYD
    TOLERANCE = 1e-5

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
        q0: np.ndarray,
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
            
        self.q0 = q0
        self.q0norm2 = np.dot(np.conjugate(self.q0), self.q0)

        self.charge_mtxel = ChargeMtxEL(
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

        self.l_g0 = self.find_g0()
        self.l_vqg = self.coulombint()

        self.l_epsinv = self.fix_eps_order()
        self.l_wcoul = self.screenedint()

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
        q0: np.ndarray,
        parallel: bool = True,
        num_bands_val: int = None,
        num_bands_con: int = None,
    ):
        kernelclass = KernelMtxEl(
            crystal=crystal,
            gspace=gspace,
            kpts=kpts,
            l_wfn=[wfn[0] for wfn in l_wfn_kgrp],
            l_gsp_wfn=[wfn[0].gkspc for wfn in l_wfn_kgrp],
            qpts=QPoints.from_cryst(kpts.recilat, epsinp.is_q0, *epsinp.qpts),
            epsinp=epsinp,
            sigmainp=sigmainp,
            l_epsmats=epsilon.l_epsinv,
            q0=q0,
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
        q0: np.ndarray,
        parallel: bool = True,
        num_bands_val: int = None,
        num_bands_con: int = None,
    ):
        l_qpts = np.array(epsinp.qpts)
        l_qpts[0] *= 0
        qpts = QPoints.from_cryst(wfndata.kpts.recilat, None, *l_qpts)

        kernelclass = KernelMtxEl(
            crystal = wfndata.crystal,
            gspace = wfndata.grho,
            kpts = wfndata.kpts,
            l_wfn = wfndata.l_wfn,
            l_gsp_wfn =  wfndata.l_gk,
            qpts = qpts, 
            epsinp = epsinp,
            sigmainp = sigmainp,
            l_epsmats = l_epsmats,
            q0=q0,
            parallel = parallel,
            num_bands_val = num_bands_val,
            num_bands_con = num_bands_con,
            )

        return kernelclass
        

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

    
    def coulombint(self):
        """
        Calculate the Coulomb potential for a given q-point, in the BGW sorted gspace.
        """

        def calc_vqg(iq: int):
            norm_array = self.l_gq[iq].gk_norm2
            sort_order = sort_cryst_like_BGW(self.l_gq[iq].gk_cryst, norm_array)

            vqg = np.where(norm_array < self.TOLERANCE, 1/self.q0norm2, 1 / np.where(norm_array < self.TOLERANCE, 1, norm_array))
            vqg = vqg[sort_order]

            return vqg

        vqg = [np.array(calc_vqg(iq)) for iq in range(self.qpts.numq)] 
        return vqg
    
    
    def fix_eps_order(self):
        """
        Fix the order of gvecs in the epsinv matrix to match other terms, in the BGW sorted eps space.
        """

        def fix_order(iq: int):
            epsinv = self.l_epsmats[iq]
            epsinv = np.conjugate(epsinv)
            norm_array = self.l_gq_epsinv[iq].gk_norm2

            sort_order = sort_cryst_like_BGW(self.l_gq_epsinv[iq].gk_cryst, norm_array)
            sort_order_QTM = np.argsort(sort_order)

            sort_order_BGW = sort_order_QTM[sort_order]
            epsinv = reorder_2d_matrix_sorted_gvecs(epsinv, sort_order_BGW)
            return epsinv
        
        l_epsinv = [np.array(fix_order(iq)) for iq in range(self.qpts.numq)]
        return l_epsinv
    
    
    def screenedint(self):
        """
        Calculate the screened Coulomb potential for a given q-point.
        """

        def calc_wqg(iq: int):
            idx_g0 = self.l_g0[iq]
            numg_eps = self.l_gq_epsinv[iq].size_g
            
            vqg = self.l_vqg[iq]
            epsinv = self.l_epsinv[iq]

            # Construct the body.
            wqg = np.diag(vqg.astype(complex))
            wqg[:numg_eps, :numg_eps] = epsinv.T * vqg[:numg_eps]

            # Treat the wings
            if iq == self.qpts.index_q0:
                wqg[idx_g0, :] = 0
                wqg[:, idx_g0] = 0
            else:
                qq = np.linalg.norm(self.qpts.cryst[iq])
                wqg[idx_g0, :numg_eps] = epsinv[:, idx_g0] * vqg[:numg_eps] * qq
                wqg[:numg_eps, idx_g0] = epsinv[idx_g0, :] * vqg[idx_g0] * qq

            # Treat the head
            wqg[idx_g0, idx_g0] = 1
            return wqg

        return [np.array(calc_wqg(iq)) for iq in range(self.qpts.numq)]

    def kernel_mtxel(self, parallel: bool = True):
        numk = self.kpts.numk
        num_val = self.charge_mtxel.val_num
        num_con = self.charge_mtxel.con_num

        def calc_kernel(ik, ikp):
            mvc = self.charge_mtxel.matrix_element("mvc", ik)
            mvcp = self.charge_mtxel.matrix_element("mvc", ikp)

            mccp = self.charge_mtxel.matrix_element("mccp", ik, ikp)
            mvvp, iq = self.charge_mtxel.matrix_element("mvvp", ik, ikp, ret_q=True)

            idx_g0 = self.l_g0[iq]
            numg_eps = self.l_gq_epsinv[iq].size_g
            wcoul = self.l_wcoul[iq]

            # Compute the head matrix elements.
            einstrh = "vV, cC -> cCvV"
            head = np.einsum(einstrh, np.conj(mvvp[..., idx_g0]), mccp[..., idx_g0], optimize=True)

            # Compute the wing matrix elements.
            einstrw1 = "vVG, G, cC -> cCvV"
            mvvp_w1 = deepcopy(mvvp)

            mvvp_w1[:, :, idx_g0] = 0
            wings = np.einsum(einstrw1, np.conj(mvvp_w1[:, :, :numg_eps]), wcoul[idx_g0, :numg_eps], mccp[:, :, idx_g0], optimize=True)

            einstrw2 = "vV, g, cCg -> cCvV"
            mccp_w2 = deepcopy(mccp)

            mccp_w2[:, :, idx_g0] = 0
            wings += np.einsum(einstrw2, np.conj(mvvp[:, :, idx_g0]), wcoul[:numg_eps, idx_g0], mccp_w2[:, :, :numg_eps], optimize=True)

            # Compute the body matrix elements.
            einstrb = "vVG, gG, cCg -> cCvV"
            mvvp_b = deepcopy(mvvp)
            mvvp_b[..., idx_g0] = 0

            mccp_b = deepcopy(mccp)
            mccp_b[..., idx_g0] = 0
            body = np.einsum(einstrb, np.conj(mvvp_b), wcoul, mccp_b, optimize=True)

            # Compute the exchange matrix elements.
            einstrx = "vcg, g, VCg -> cCvV"
            vq0g = deepcopy(self.l_vqg[self.qpts.index_q0])

            vq0g[0] = 0
            exc = np.einsum(einstrx, np.conj(mvc), vq0g, mvcp, optimize=True)

            del mvvp_w1, mvvp_b, mccp_w2, mccp_b, vq0g

            return exc, head, wings, body

        if self.in_parallel and parallel:
            proc_rank = self.comm.Get_rank()

            if proc_rank == 0:
                full_pairs = [(k_idx, kp_idx) for k_idx in range(numk) for kp_idx in range(numk) if k_idx <= kp_idx]

                full_pairs = np.array(full_pairs)
                chunks = np.array_split(full_pairs, self.comm.Get_size())

            else:
                chunks = None

            local_pairs = self.comm.scatter(chunks, root=0)
            local_blocks = []

            for pair in local_pairs:
                ik, ikp = pair
                exc, head, wings, body = calc_kernel(ik, ikp)

                local_blocks.append((ik, ikp, exc, head, wings, body))
                
                if ik != ikp:
                    local_blocks.append((ikp, ik, np.conjugate(exc.transpose(1, 0, 3, 2)), np.conjugate(head.transpose(1, 0, 3, 2)), np.conjugate(wings.transpose(1, 0, 3, 2)), np.conjugate(body.transpose(1, 0, 3, 2))))

            if proc_rank != 0:
                self.comm.send(local_blocks, dest=0, tag=77)
                exc_result = self.comm.bcast(None, root=0)

                head_result = self.comm.bcast(None, root=0)
                wings_result = self.comm.bcast(None, root=0)
                body_result = self.comm.bcast(None, root=0)

            else:
                exc_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)
                head_mtx = np.zeros_like(exc_mtx)

                wings_mtx = np.zeros_like(exc_mtx)
                body_mtx = np.zeros_like(exc_mtx)

                for ik_idx, ikp_idx, exc, head, wings, body in local_blocks:
                    exc_mtx[ik_idx, ikp_idx] = exc
                    head_mtx[ik_idx, ikp_idx] = head

                    wings_mtx[ik_idx, ikp_idx] = wings
                    body_mtx[ik_idx, ikp_idx] = body

                for source in range(1, self.comm_size):
                    remote_blocks = self.comm.recv(source=source, tag=77)

                    for ik_idx, ikp_idx, exc, head, wings, body in remote_blocks:
                        exc_mtx[ik_idx, ikp_idx] = exc

                        head_mtx[ik_idx, ikp_idx] = head
                        wings_mtx[ik_idx, ikp_idx] = wings
                        body_mtx[ik_idx, ikp_idx] = body


                exc_result = exc_mtx
                head_result = head_mtx

                wings_result = wings_mtx
                body_result = body_mtx

                self.comm.bcast(exc_result, root=0)
                self.comm.bcast(head_result, root=0)

                self.comm.bcast(wings_result, root=0)
                self.comm.bcast(body_result, root=0)

            return {"exc": exc_result, "head": head_result, "wings": wings_result, "body": body_result,}
        
        else:
            exc_mtx = np.zeros((numk, numk, num_con, num_con, num_val, num_val), dtype=complex)
            head_mtx = np.zeros_like(exc_mtx)

            wings_mtx = np.zeros_like(exc_mtx)
            body_mtx = np.zeros_like(exc_mtx)

            for ikp_idx in range(numk):
                for ik_idx in range(ikp_idx + 1):
                    exc, head, wings, body = calc_kernel(ik_idx, ikp_idx)

                    exc_mtx[ik_idx, ikp_idx] = exc
                    head_mtx[ik_idx, ikp_idx] = head

                    wings_mtx[ik_idx, ikp_idx] = wings
                    body_mtx[ik_idx, ikp_idx] = body

                    if ik_idx != ikp_idx:
                        exc_mtx[ikp_idx, ik_idx] = np.conjugate(exc.transpose(1, 0, 3, 2))
                        head_mtx[ikp_idx, ik_idx] = np.conjugate(head.transpose(1, 0, 3, 2))

                        wings_mtx[ikp_idx, ik_idx] = np.conjugate(wings.transpose(1, 0, 3, 2))
                        body_mtx[ikp_idx, ik_idx] = np.conjugate(body.transpose(1, 0, 3, 2))

            return {"exc": exc_mtx, "head": head_mtx, "wings": wings_mtx, "body": body_mtx}
                

