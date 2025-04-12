### IDEA TO FIX EPSINV vs NORMAL INDEXING:
### Create two mccp, mvvp, mvc - one for normal large gq space and the other for epsinv gq space.
### Calculate the head, wings and exchange using the epsinv gq space. For the unscreened part
### use the portion of the large gq space that is not in the epsinv gq space.


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
                    ecutwfn=self.epsinp.epsilon_cutoff * RYDBERG_HART,
                )
            )
        
        # Store the index of G = 0 for each q-point.
        idxG0 = np.zeros(self.qpts.numq, dtype=int)
        for q_idx in range(self.qpts.numq):
            sort_order = sort_cryst_like_BGW(
                self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2
            )
            idxG0[q_idx] = sort_order[0]
        self.idxG0 = idxG0    

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

        # NOTE: sorting is unclear
        self.l_epsinv = []
        for i_q in range(self.qpts.numq):
            epsinv = self.l_epsmats[i_q]

            sort_order = sort_cryst_like_BGW(
                self.l_gq_epsinv[i_q].gk_cryst, self.l_gq_epsinv[i_q].gk_norm2
            )

            self.l_epsinv.append(
                reorder_2d_matrix_sorted_gvecs(epsinv, np.argsort(sort_order))
            )
            
        self.q0val = q0val
            
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
            q0val=q0val,
            parallel = parallel,
            num_bands_val = num_bands_val,
            num_bands_con = num_bands_con,
            )

        return kernelclass
    
    # Define helper method to calculate the head of the direct kernel matrix.
    def calc_head(
            self,
            idx_G0,
            vqg,    # Value of v(q + G) for q
            epsinv, # Inverse of epsilon matrix for q
            mccp,   # Charge matrix element for valence bands, k = k_idx, k' = kp_idx
            mvvp,   # Charge matrix element for conduction bands, k = k_idx, k' = kp_idx
    ):
        # NOTE: epsinv lives in a smaller gspace, due to the epsinp cutoff.
        # So, we set up everything in the same gspace as epsinv.
        gsize = epsinv.shape[0]

        # Set up parameters for head with G = 0, G' = 0.
        # As in BerkeleyGW, we set W(G = 0, G' = 0) = 1 for semiconductor screening.

        mccp_head = deepcopy(mccp) # mccp -> (c, C, g)
        mccp_head = mccp_head[..., :gsize] # mccp -> (c, C, g)
        mccp_head = mccp_head[..., idx_G0] # mccp -> (c, C, g = 0)

        mvvp_head = deepcopy(mvvp) # mvvp -> (v, V, G)
        mvvp_head = mvvp_head[..., :gsize] # mvvp -> (v, V, G)
        mvvp_head = mvvp_head[..., idx_G0] # mvvp -> (v, V, G = 0)

        # Head = sum(g=0, G=0)[conj(mvvp(G)) * W(g, G) * mccp(g)]
        einstr = "vV, cC -> cCvV"
        head = np.einsum(
            einstr,
            np.conj(mvvp_head),
            mccp_head,
            optimize=True,
        )

        # Remove stored deepcopies after calculation
        del mccp_head
        del mvvp_head

        return head
    
    # Define helper method to calculate the wings of the direct kernel matrix.
    def calc_wings(
            self,
            idx_G0,
            q_idx,  # Index of q-point
            vqg,    # Value of v(q + G) for q
            epsinv, # Inverse of epsilon matrix for q
            mccp,   # Charge matrix element for valence bands, k = k_idx, k' = kp_idx
            mvvp,   # Charge matrix element for conduction bands, k = k_idx, k' = kp_idx
    ):

        # NOTE: epsinv lives in a smaller gspace, due to the epsinp cutoff.
        # So, we set up everything in the same gspace as epsinv.
        gsize = epsinv.shape[0]

        # Set up parameters for wing with G = 0, G' != 0 -> wing prime.         
        mccp_wingp = deepcopy(mccp) # mccp -> (c, C, g)
        mccp_wingp = mccp_wingp[..., :gsize] # mccp -> (c, C, g)
        mccp_wingp = mccp_wingp[..., idx_G0] # mccp -> (c, C, g = 0)

        mvvp_wingp = deepcopy(mvvp) # mvvp -> (v, V, G)
        mvvp_wingp = mvvp_wingp[..., :gsize] # mvvp -> (v, V, G)
        mvvp_wingp[..., idx_G0] = 0 # mvvp -> (v, V, G != 0)

        vqg_wingp = deepcopy(vqg) # vqg -> G
        vqg_wingp = vqg_wingp[:gsize] # vqg -> G
        vqg_wingp[idx_G0] = 0 # vqg -> G != 0

        epsinv_wingp = deepcopy(epsinv) # epsinv -> (g, G)
        epsinv_wingp = epsinv_wingp[idx_G0, :] # epsinv -> (g = 0, G)
        epsinv_wingp[idx_G0] = 0 # epsinv -> (g = 0, G != 0)

        # wing prime = sum(g=0, G != 0)[conj(mvvp(G)) * espinv(g, G) * vqg(G) * mccp(g)]
        einstr_wingp = "vVG, G, G, cC -> cCvV"
        wingp = np.einsum(
            einstr_wingp,
            np.conj(mvvp_wingp),
            epsinv_wingp,
            vqg_wingp,
            mccp_wingp,
            optimize=True,
        )

        # Remove stored deepcopies after calculation
        del mccp_wingp
        del mvvp_wingp

        del vqg_wingp
        del epsinv_wingp
        
        # Set up parameters for wing with G != 0, G' = 0. -> wing
        mccp_wing = deepcopy(mccp) # mccp -> (c, C, g)
        mccp_wing = mccp_wing[..., :gsize] # mccp -> (c, C, g)
        mccp_wing[..., idx_G0] = 0 # mccp -> (c, C, g != 0)

        mvvp_wing = deepcopy(mvvp) # mvvp -> (v, V, G)
        mvvp_wing = mvvp_wing[..., :gsize] # mvvp -> (v, V, G)
        mvvp_wing = mvvp_wing[..., idx_G0] # mvvp -> (v, V, G = 0)

        vqg_wing = deepcopy(vqg) # vqg -> G
        vqg_wing = vqg_wing[:gsize] # vqg -> G
        vqg_wing = vqg_wing[idx_G0] # vqg -> G = 0

        epsinv_wing = deepcopy(epsinv) # epsinv -> (g, G)
        epsinv_wing = epsinv_wing[:, idx_G0] # epsinv -> (g, G = 0)
        epsinv_wing[idx_G0] = 0 # epsinv -> (g != 0, G = 0)

        # wing = sum(g != 0, G = 0)[conj(mvvp(G)) * espinv(g, G) * vqg(G) * mccp(g)]
        einstr_wing = "vV, g, cCg -> cCvV"
        wing = np.einsum(
            einstr_wing,
            np.conj(mvvp_wing),
            epsinv_wing,
            mccp_wing,
            optimize=True,
        )

        wing = wing * vqg_wing
        # Remove stored deepcopies after calculation
        del mccp_wing
        del mvvp_wing

        del vqg_wing
        del epsinv_wing

        wings = wing + wingp

        if q_idx == self.qpts.index_q0:
            wings = np.zeros_like(wings)
        else:
            qval = np.linalg.norm(self.qpts.cryst[q_idx])
            wings = wings * qval

        return wings
    
    # Define helper method to calculate the body of the direct kernel matrix.
    def calc_body(
            self,
            idx_G0,
            vqg,    # Value of v(q + G) for q
            epsinv, # Inverse of epsilon matrix for q
            mccp,   # Charge matrix element for valence bands, k = k_idx, k' = kp_idx
            mvvp,   # Charge matrix element for conduction bands, k = k_idx, k' = kp_idx            
    ):
        # NOTE: epsinv lives in a smaller gspace, due to the epsinp cutoff.
        # So, we set up everything in the same gspace as epsinv.
        gsize = epsinv.shape[0]

        # Set up parameters for body with G != 0, G' != 0.
        mccp_body = deepcopy(mccp) # mccp -> (c, C, g)
        mccp_body = mccp_body[..., :gsize] # mccp -> (c, C, g)
        mccp_body[..., idx_G0] = 0 # mccp -> (c, C, g != 0)

        mvvp_body = deepcopy(mvvp) # mvvp -> (v, V, G)
        mvvp_body = mvvp_body[..., :gsize] # mvvp -> (v, V, G)
        mvvp_body[..., idx_G0] = 0 # mvvp -> (v, V, G != 0)

        vqg_body = deepcopy(vqg) # vqg -> G
        vqg_body = vqg_body[:gsize] # vqg -> G
        vqg_body[idx_G0] = 0

        epsinv_body = deepcopy(epsinv) # epsinv -> (g, G)
        epsinv_body[idx_G0, idx_G0] = 0 # epsinv -> (g != 0, G != 0)

        # body = sum(g != 0, G != 0)[conj(mvvp(G)) * espinv(g, G) * vqg(G) * mccp(g)]
        einstr_body = "vVG, gG, G, cCg -> cCvV"
        body = np.einsum(
            einstr_body,
            np.conj(mvvp_body),
            epsinv_body,
            vqg_body,
            mccp_body,
            optimize=True,
        )

        # Remove stored deepcopies after calculation
        del mccp_body
        del mvvp_body
        del vqg_body
        del epsinv_body

        # Add contributions from unscreened coulomb interaction at "high" G.
        mccp_usc = deepcopy(mccp) # mccp -> (c, C, g)
        mccp_usc = mccp_usc[..., gsize:] # mccp -> (c, C, g > g_eps)

        mvvp_usc = deepcopy(mvvp) # mvvp -> (v, V, G)
        mvvp_usc = mvvp_usc[..., gsize:] # mvvp -> (v, V, G > g_eps)

        vqg_usc = deepcopy(vqg) # vqg -> G
        vqg_usc = vqg_usc[gsize:] # vqg -> G > g_eps

        # body_usc = sum(g > g_eps)[conj(mvvp(g)) * vqg(g) * mccp(g)]
        einstr_body_usc = "vVg, g, cCg -> cCvV"
        body_usc = np.einsum(
            einstr_body_usc,
            np.conj(mvvp_usc),
            vqg_usc,
            mccp_usc,
            optimize=True,
        )

        # Remove stored deepcopies after calculation
        del mccp_usc
        del mvvp_usc    
        del vqg_usc

        # Add the unscreened contribution to the body.
        body += body_usc

        return body

    # Define helper method to calculate the exchange kernel matrix.
    def calc_exc(
            self,
            vq0g, # Value of v(q + G) for q = 0.
            mvc,  # Charge matrix element mvc for k = k_idx.
            mvcp, # Charge matrix element mvc for k' = kp_idx.            
    ):  
        # Set v(q = 0, G = 0) to 0.
        vq0g_exc = deepcopy(vq0g)
        vq0g_exc[0] = 0.0

        # Shape required for the result is (v,V,c,C).
        # mvc -> (v,c,g), vq0g -> g & mvcp -> (V, C, g)
        einstr = "vcg, g, VCg -> cCvV"

        exc = np.einsum(
            einstr,
            np.conj(mvc),
            vq0g_exc,
            mvcp,
            optimize=True,
        )

        return exc
    
    def calc_kernel_mtxel(self, parallel: bool = True):
        numk = self.kpts.numk
        num_val = self.charge_mtxel.val_num
        num_con = self.charge_mtxel.con_num

        # --- Calculate the value of v(q + G) for q = 0 ---
        norm_array = self.l_gq[0].gk_norm2
        vq0g = np.where(norm_array == 0, 0, 1 / np.where(norm_array == 0, 1, norm_array))
        # sort_order_0 = sort_cryst_like_BGW(
        #     self.l_gq[0].gk_cryst, norm_array
        # )
        # vq0g = vq0g[sort_order_0]

        # If using parallel, work with MPI; otherwise run serially.
        if self.in_parallel and parallel:
            proc_rank = self.comm.Get_rank()

            # On the root process, generate the list of unique pairs with k >= kp.
            if proc_rank == 0:
                full_pairs = [
                    (k_idx, kp_idx)
                    for k_idx in range(numk)
                    for kp_idx in range(numk)
                    if k_idx >= kp_idx
                ]
                full_pairs = np.array(full_pairs)
                chunks = np.array_split(full_pairs, self.comm.Get_size())
            else:
                chunks = None

            # Scatter the unique k–kp pairs among all processes.
            local_pairs = self.comm.scatter(chunks, root=0)

            # Compute local blocks for each k–kp pair in the received chunk.
            local_blocks = []
            for pair in local_pairs:
                k_idx, kp_idx = pair

                # Retrieve quantities for the given k-point pair.
                mvc = self.charge_mtxel.mvc(k_idx)
                mvcp = self.charge_mtxel.mvc(kp_idx)

                data_mvvp = self.charge_mtxel.mvvp(
                    k_idx=k_idx,
                    kp_idx=kp_idx,
                    ret_q='True'
                )
                mvvp = data_mvvp[0]
                q_idx = data_mvvp[1]

                mccp = self.charge_mtxel.mccp(
                    k_idx=k_idx,
                    kp_idx=kp_idx,
                )

                # Calculate v(q+G) for this q-index.
                norm_array = self.l_gq[q_idx].gk_norm2
                vqg = np.where(norm_array == 0, 0, 1 / np.where(norm_array == 0, 1, norm_array))
                # sort_order = sort_cryst_like_BGW(
                #     self.l_gq[q_idx].gk_cryst, norm_array
                # )
                # vqg = vqg[sort_order]

                idx_G0 = self.idxG0[q_idx]

                # epsinv was already sorted.
                epsinv = self.l_epsinv[q_idx]

                # Calculate the different components of the kernel.
                head = self.calc_head(
                    idx_G0=idx_G0,
                    vqg=vqg,
                    epsinv=epsinv,
                    mccp=mccp,
                    mvvp=mvvp,
                )
                wings = self.calc_wings(
                    idx_G0=idx_G0,
                    q_idx=q_idx,
                    vqg=vqg,
                    epsinv=epsinv,
                    mccp=mccp,
                    mvvp=mvvp,
                )
                body = self.calc_body(
                    idx_G0=idx_G0,
                    vqg=vqg,
                    epsinv=epsinv,
                    mccp=mccp,
                    mvvp=mvvp,
                )
                exc = self.calc_exc(
                    vq0g=vq0g,
                    mvc=mvc,
                    mvcp=mvcp,
                )
                local_blocks.append((k_idx, kp_idx, exc, head, wings, body))

            # Communication after local computation.
            if proc_rank != 0:
                self.comm.send(local_blocks, dest=0, tag=77)
                exc_result = self.comm.bcast(None, root=0)
                head_result = self.comm.bcast(None, root=0)
                wings_result = self.comm.bcast(None, root=0)
                body_result = self.comm.bcast(None, root=0)
            else:
                # Initialize empty full matrices.
                exc_mtx = np.zeros(
                    (numk, numk, num_con, num_con, num_val, num_val),
                    dtype=complex
                )
                head_mtx = np.zeros(
                    (numk, numk, num_con, num_con, num_val, num_val),
                    dtype=complex
                )
                wings_mtx = np.zeros(
                    (numk, numk, num_con, num_con, num_val, num_val),
                    dtype=complex
                )
                body_mtx = np.zeros(
                    (numk, numk, num_con, num_con, num_val, num_val),
                    dtype=complex
                )

                # Place the computed results from the root's own data.
                for k_idx, kp_idx, exc, head, wings, body in local_blocks:
                    exc_mtx[k_idx, kp_idx] = exc
                    head_mtx[k_idx, kp_idx] = head
                    wings_mtx[k_idx, kp_idx] = wings
                    body_mtx[k_idx, kp_idx] = body

                # Collect the computed blocks from the other processes.
                for source in range(1, self.comm_size):
                    remote_blocks = self.comm.recv(source=source, tag=77)
                    for k_idx, kp_idx, exc, head, wings, body in remote_blocks:
                        exc_mtx[k_idx, kp_idx] = exc
                        head_mtx[k_idx, kp_idx] = head
                        wings_mtx[k_idx, kp_idx] = wings
                        body_mtx[k_idx, kp_idx] = body

                # --- Symmetrize the matrices ---
                # Fill in the missing half (k < kp) by symmetry.
                for k_idx in range(numk):
                    for kp_idx in range(k_idx):
                        # In this symmetric case, the element (k, kp) equals (kp, k).
                        exc_mtx[kp_idx, k_idx] = exc_mtx[k_idx, kp_idx]
                        head_mtx[kp_idx, k_idx] = head_mtx[k_idx, kp_idx]
                        wings_mtx[kp_idx, k_idx] = wings_mtx[k_idx, kp_idx]
                        body_mtx[kp_idx, k_idx] = body_mtx[k_idx, kp_idx]

                exc_result = exc_mtx
                head_result = head_mtx
                wings_result = wings_mtx
                body_result = body_mtx

                # Broadcast the full results to all processes.
                self.comm.bcast(exc_result, root=0)
                self.comm.bcast(head_result, root=0)
                self.comm.bcast(wings_result, root=0)
                self.comm.bcast(body_result, root=0)

            return {
                "exc": exc_result,
                "head": head_result,
                "wings": wings_result,
                "body": body_result,
            }

        else:
            # --- Serial (non-parallel) implementation ---
            exc_mtx = np.zeros(
                (numk, numk, num_con, num_con, num_val, num_val),
                dtype=complex,
            )
            head_mtx = np.zeros(
                (numk, numk, num_con, num_con, num_val, num_val),
                dtype=complex,
            )
            wings_mtx = np.zeros(
                (numk, numk, num_con, num_con, num_val, num_val),
                dtype=complex,
            )
            body_mtx = np.zeros(
                (numk, numk, num_con, num_con, num_val, num_val),
                dtype=complex,
            )

            # Loop only over half the pairs, where k_idx >= kp_idx.
            for k_idx in range(numk):
                for kp_idx in range(k_idx + 1):  # ensures k_idx >= kp_idx
                    mvc = self.charge_mtxel.mvc(k_idx)
                    mvcp = self.charge_mtxel.mvc(kp_idx)

                    data_mvvp = self.charge_mtxel.mvvp(
                        k_idx=k_idx,
                        kp_idx=kp_idx,
                        ret_q='True'
                    )
                    mvvp = data_mvvp[0]
                    q_idx = data_mvvp[1]

                    mccp = self.charge_mtxel.mccp(
                        k_idx=k_idx,
                        kp_idx=kp_idx,
                    )

                    norm_array = self.l_gq[q_idx].gk_norm2
                    vqg = np.where(norm_array == 0, 0, 1 / np.where(norm_array == 0, 1, norm_array))
                    # sort_order = sort_cryst_like_BGW(
                    #     self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2
                    # )
                    # vqg = vqg[sort_order]

                    idx_G0 = self.idxG0[q_idx]

                    epsinv = self.l_epsinv[q_idx]

                    head = self.calc_head(
                        idx_G0=idx_G0,
                        vqg=vqg,
                        epsinv=epsinv,
                        mccp=mccp,
                        mvvp=mvvp,
                    )
                    wings = self.calc_wings(
                        idx_G0=idx_G0,
                        q_idx=q_idx,
                        vqg=vqg,
                        epsinv=epsinv,
                        mccp=mccp,
                        mvvp=mvvp,
                    )
                    body = self.calc_body(
                        idx_G0=idx_G0,
                        vqg=vqg,
                        epsinv=epsinv,
                        mccp=mccp,
                        mvvp=mvvp,
                    )
                    exc = self.calc_exc(
                        vq0g=vq0g,
                        mvc=mvc,
                        mvcp=mvcp,
                    )

                    exc_mtx[k_idx, kp_idx] = exc
                    head_mtx[k_idx, kp_idx] = head
                    wings_mtx[k_idx, kp_idx] = wings
                    body_mtx[k_idx, kp_idx] = body

            # --- Symmetrize the matrices ---
            for k_idx in range(numk):
                for kp_idx in range(k_idx):
                    exc_mtx[kp_idx, k_idx] = exc_mtx[k_idx, kp_idx]
                    head_mtx[kp_idx, k_idx] = head_mtx[k_idx, kp_idx]
                    wings_mtx[kp_idx, k_idx] = wings_mtx[k_idx, kp_idx]
                    body_mtx[kp_idx, k_idx] = body_mtx[k_idx, kp_idx]

            return {
                "exc": exc_mtx,
                "head": head_mtx,
                "wings": wings_mtx,
                "body": body_mtx,
            }


    
    # def calc_kernel_mtxel(
    #         self,
    #         parallel:bool = True,
    # ):
    #     numk = self.kpts.numk

    #     num_val = self.charge_mtxel.val_num
    #     num_con = self.charge_mtxel.con_num

    #     # Calculate the value of v(q + G) for q = 0.
    #     norm_array = self.l_gq[0].gk_norm2
    #     vq0g = np.where(norm_array == 0, 0, 1 / np.where(norm_array == 0, 1, norm_array)) 

    #     # Sort the indices of v(q + G), for q = 0.
    #     sort_order_0 = sort_cryst_like_BGW(
    #         self.l_gq[0].gk_cryst, norm_array
    #     )

    #     vq0g = vq0g[sort_order_0]

    #     if self.in_parallel and parallel:
    #         proc_rank = self.comm.Get_rank()

    #         if proc_rank == 0:
    #             k_indices, kp_indices = np.meshgrid(
    #                 np.arange(numk),
    #                 np.arange(numk),
    #                 indexing="ij",
    #             )
    #             full_pairs = np.stack(
    #                 (k_indices.flatten(), kp_indices.flatten()), axis=1
    #             )
    #             chunks = np.array_split(full_pairs, self.comm.Get_size())
    #         else:
    #             chunks = None

    #         local_pairs = self.comm.scatter(chunks, root=0)

    #         # Each element: (k_idx, kp_idx, exc_block, head_block, wings_block, body_block)
    #         local_blocks = []  

    #         for pair in local_pairs:
    #             k_idx, kp_idx = pair

    #             mvc = self.charge_mtxel.mvc(k_idx)

    #             mvcp = self.charge_mtxel.mvc(kp_idx)

    #             data_mvvp = self.charge_mtxel.mvvp(
    #                 k_idx = k_idx,
    #                 kp_idx = kp_idx,
    #                 ret_q = 'True'
    #             )

    #             mvvp = data_mvvp[0]
    #             q_idx = data_mvvp[1]

    #             mccp = self.charge_mtxel.mccp(
    #                 k_idx = k_idx,
    #                 kp_idx = kp_idx,
    #             )

    #             # Calculate the value of v(q + G).
    #             norm_array = self.l_gq[q_idx].gk_norm2
    #             vqg = np.where(norm_array == 0, 0, 1 / np.where(norm_array == 0, 1, norm_array))

    #             # Sort the indices of v(q + G).
    #             sort_order = sort_cryst_like_BGW(
    #                 self.l_gq[q_idx].gk_cryst, norm_array
    #             )
    #             vqg = vqg[sort_order]

    #             # Indices of epsinv were already sorted.
    #             epsinv = self.l_epsinv[q_idx]

    #             # Calculate the head, wings, body and exchange part of the kernel.
    #             head = self.calc_head(
    #                 vqg = vqg,
    #                 epsinv = epsinv,
    #                 mccp = mccp,
    #                 mvvp = mvvp,
    #             )

    #             wings = self.calc_wings(
    #                 q_idx = q_idx,
    #                 vqg = vqg,
    #                 epsinv = epsinv,
    #                 mccp = mccp,
    #                 mvvp = mvvp,
    #             )

    #             body = self.calc_body(
    #                 vqg = vqg,
    #                 epsinv = epsinv,
    #                 mccp = mccp,
    #                 mvvp = mvvp,
    #             )

    #             exc = self.calc_exc(
    #                 vq0g = vq0g,
    #                 mvc = mvc,
    #                 mvcp = mvcp,
    #             )

    #             local_blocks.append(
    #                 (k_idx, kp_idx, exc, head, wings, body)
    #             )
            
    #         if proc_rank != 0:
    #             self.comm.send(local_blocks, dest=0, tag=77)

    #             exc_result = self.comm.bcast(None, root = 0)
    #             head_result = self.comm.bcast(None, root = 0)

    #             wings_result = self.comm.bcast(None, root = 0)
    #             body_result = self.comm.bcast(None, root = 0)

    #         else:
    #             exc_mtx = np.zeros(
    #                 (numk, numk, num_con, num_con, num_val, num_val),
    #                 dtype = complex
    #             )

    #             head_mtx = np.zeros(
    #                 (numk, numk, num_con, num_con, num_val, num_val),
    #                 dtype = complex
    #             )

    #             wings_mtx = np.zeros(
    #                 (numk, numk, num_con, num_con, num_val, num_val),
    #                 dtype = complex
    #             )

    #             body_mtx = np.zeros(
    #                 (numk, numk, num_con, num_con, num_val, num_val),
    #                 dtype = complex
    #             )

    #             for k_idx, kp_idx, exc, head, wings, body in local_blocks:
    #                 exc_mtx[k_idx, kp_idx] += exc
    #                 head_mtx[k_idx, kp_idx] += head

    #                 wings_mtx[k_idx, kp_idx] += wings
    #                 body_mtx[k_idx, kp_idx] += body

    #             for source in range(1, self.comm_size):
    #                 remote_blocks = self.comm.recv(source=source, tag=77)

    #                 for k_idx, kp_idx, exc, head, wings, body in remote_blocks:
    #                     exc_mtx[k_idx, kp_idx] += exc
    #                     head_mtx[k_idx, kp_idx] += head

    #                     wings_mtx[k_idx, kp_idx] += wings
    #                     body_mtx[k_idx, kp_idx] += body

    #             exc_result = exc_mtx
    #             head_result = head_mtx

    #             wings_result = wings_mtx
    #             body_result = body_mtx

    #             # Broadcast the results to all processes.
    #             self.comm.bcast(exc_result, root=0)
    #             self.comm.bcast(head_result, root=0)

    #             self.comm.bcast(wings_result, root=0)
    #             self.comm.bcast(body_result, root=0)

    #         return {
    #             "exc": exc_result,
    #             "head": head_result,
    #             "wings": wings_result,
    #             "body": body_result,
    #         }

    #     else:
    #         exc_result = np.zeros(
    #             (numk, numk, num_con, num_con, num_val, num_val),
    #             dtype=complex,
    #         )

    #         head_result = np.zeros(
    #             (numk, numk, num_con, num_con, num_val, num_val),
    #             dtype=complex,
    #         )

    #         wings_result = np.zeros(
    #             (numk, numk, num_con, num_con, num_val, num_val),
    #             dtype=complex,
    #         )

    #         body_result = np.zeros(
    #             (numk, numk, num_con, num_con, num_val, num_val),
    #             dtype=complex,
    #         )

    #         for k_idx in range(numk):
    #             for kp_idx in range(numk):

    #                 mvc = self.charge_mtxel.mvc(k_idx)
    #                 mvcp = self.charge_mtxel.mvc(kp_idx)

    #                 data_mvvp = self.charge_mtxel.mvvp(
    #                     k_idx=k_idx,
    #                     kp_idx=kp_idx,
    #                     ret_q='True'
    #                 )

    #                 mvvp = data_mvvp[0]
    #                 q_idx = data_mvvp[1]

    #                 mccp = self.charge_mtxel.mccp(
    #                     k_idx=k_idx,
    #                     kp_idx=kp_idx,
    #                 )

    #                 # Calculate the value of v(q + G).
    #                 norm_array = self.l_gq[q_idx].gk_norm2
    #                 vqg = np.where(norm_array == 0, 0, 1 / np.where(norm_array == 0, 1, norm_array))
                    
    #                 # Sort the indices of v(q + G).
    #                 sort_order = sort_cryst_like_BGW(
    #                     self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2
    #                 )
    #                 vqg = vqg[sort_order]

    #                 # Indices of epsinv were already sorted.
    #                 epsinv = self.l_epsinv[q_idx]

    #                 # Calculate the head, wings and body of the kernel.
    #                 head = self.calc_head(
    #                     vqg=vqg,
    #                     epsinv=epsinv,
    #                     mccp=mccp,
    #                     mvvp=mvvp,
    #                 )

    #                 wings = self.calc_wings(
    #                     q_idx=q_idx,
    #                     vqg=vqg,
    #                     epsinv=epsinv,
    #                     mccp=mccp,
    #                     mvvp=mvvp,
    #                 )

    #                 body = self.calc_body(
    #                     vqg=vqg,
    #                     epsinv=epsinv,
    #                     mccp=mccp,
    #                     mvvp=mvvp,
    #                 )

    #                 exc = self.calc_exc(
    #                     vq0g=vq0g,
    #                     mvc=mvc,
    #                     mvcp=mvcp,
    #                 )

    #                 exc_result[k_idx, kp_idx] += exc
    #                 head_result[k_idx, kp_idx] += head

    #                 wings_result[k_idx, kp_idx] += wings
    #                 body_result[k_idx, kp_idx] += body

    #         return {
    #             "exc": exc_result,
    #             "head": head_result,
    #             "wings": wings_result,
    #             "body": body_result,
    #         }
        










