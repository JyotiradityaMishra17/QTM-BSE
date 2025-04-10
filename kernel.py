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

        self.vcoul = Vcoul(
            gspace=self.gspace,
            qpts=self.qpts,
            bare_coulomb_cutoff=sigmainp.bare_coulomb_cutoff,
            parallel=self.in_parallel,
        )

        # self.vcoul.calculate_vcoul(averaging_func=self.vcoul.v_minibz_montecarlo_hybrid)
        self.l_epsinv = []
        for i_q in range(self.qpts.numq):
            epsinv = self.l_epsmats[i_q]
            # if self.fixwings:
            #     epsinv = self.vcoul.calculate_fixedeps(
            #         epsinv, i_q, random_sample=False
            #     )
            sort_order = sort_cryst_like_BGW(
                self.l_gq[i_q].gk_cryst, self.l_gq[i_q].gk_norm2

            )
            self.l_epsinv.append(
                reorder_2d_matrix_sorted_gvecs(epsinv, sort_order)
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

        # self.kernel_factor = KernelMtxEl.ryd / ((crystal.reallat.cellvol) * self.qpts.numq)

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
            q_idx,  # Index of q-point
            vqg,    # Value of v(q + G) for q
            epsinv, # Inverse of epsilon matrix for q
            mccp,   # Charge matrix element for valence bands, k = k_idx, k' = kp_idx
            mvvp,   # Charge matrix element for conduction bands, k = k_idx, k' = kp_idx
    ):
        # Set up parameters for head with G = 0, G' = 0.
        epsinv_0_0 = np.zeros_like(epsinv)
        epsinv_0_0[0, 0] = 1 # epsinv -> (g = 0, G = 0)

        # Shape required for head is (v,V,c,C).
        # mccp -> (c, C, g), epsinv -> (g = 0, G = 0), vqg -> (G), mvvp -> (v, V, G)
        einstr = 'cCg, gG, G, vVG -> vVcC'
        head = np.einsum(
            einstr,
            mccp,
            epsinv_0_0,
            np.ones_like(vqg),
            np.conj(mvvp),
            optimize=True,
        )

        return head
    
    # Define helper method to calculate the wings of the direct kernel matrix.
    def calc_wings(
            self,
            q_idx,  # Index of q-point
            vqg,    # Value of v(q + G) for q
            epsinv, # Inverse of epsilon matrix for q
            mccp,   # Charge matrix element for valence bands, k = k_idx, k' = kp_idx
            mvvp,   # Charge matrix element for conduction bands, k = k_idx, k' = kp_idx
    ):
        # Set up parameters for wing with G = 0, G' != 0.
        mccp_local = np.zeros_like(mccp)
        mccp_local[..., 0] = mccp[..., 0] # mccp -> (c, C, g = 0)

        epsinv_0_n0 = deepcopy(epsinv)
        epsinv_0_n0[:, 0] = 0.0

        # Shape required for wing1 is (v,V,c,C).
        # mccp -> (c, C, g = 0), epsinv -> (g, G != 0), vqg -> G, mvvp -> (v, V, G)
        einstr = 'cCg, gG, G, vVG -> vVcC'
        wing = np.einsum(
            einstr,
            mccp_local,
            epsinv_0_n0,
            vqg,
            np.conj(mvvp),
            optimize=True,
        )

        # Set up parameters for wing with G != 0, G' = 0.
        mvvp_local = np.zeros_like(mvvp)
        mvvp_local[..., 0] = mvvp[..., 0] # mvvp -> (v, V, G = 0)

        epsinv_n0_0 = deepcopy(epsinv)
        epsinv_n0_0[0, :] = 0.0

        wing += np.einsum(
            einstr,
            mccp,
            epsinv_n0_0,
            vqg,
            np.conj(mvvp_local),
            optimize=True,
        )

        if q_idx == self.qpts.index_q0:
            wing *= 0

        else:
            oneoverq = self.vcoul.oneoverq[q_idx]
            qval = np.sqrt(
                self.l_gq[q_idx].cryst_to_norm2(self.qpts.cryst[q_idx])
            )

            # wing *= oneoverq * qval / (8 * np.pi)
            wing *= qval

        return wing #* self.kernel_factor
    
    # Define helper method to calculate the body of the direct kernel matrix.
    # We need (G != 0, G' != 0) for the body.
    def calc_body(
            self,
            vqg,    # Value of v(q + G) for q
            epsinv, # Inverse of epsilon matrix for q
            mccp,   # Charge matrix element for valence bands, k = k_idx, k' = kp_idx
            mvvp,   # Charge matrix element for conduction bands, k = k_idx, k' = kp_idx            
    ):

        epsinv_n0_n0 = deepcopy(epsinv)
        epsinv_n0_n0[0, :] = 0.0
        epsinv_n0_n0[:, 0] = 0.0

        # Shape required for the result is (v,V,c,C).
        # mccp -> (c, C, g), epsinv -> (g != 0, G != 0), vqg -> G, mvvp -> (v, V, G)
        einstr = 'cCg, gG, G, vVG -> vVcC'

        body = np.einsum(
            einstr,
            mccp,
            epsinv_n0_n0,
            vqg,
            np.conj(mvvp),
            optimize=True,
        )

        return body #* self.kernel_factor

    # Define helper method to calculate the exchange kernel matrix.
    def calc_exc(
            self,
            vq0g, # Value of v(q + G) for q = 0.
            mvc,  # Charge matrix element mvc for k = k_idx.
            mvcp, # Charge matrix element mvc for k' = kp_idx.            
    ):  


        # Set v(q = 0, G = 0) to 0.
        vq0g_local = deepcopy(vq0g)
        vq0g_local[0] = 0.0

        # Shape required for the result is (v,V,c,C).
        # mvc -> (v,c,g), vq0g -> g & mvcp -> (V, C, g)
        einstr = "vcg, g, VCg -> vVcC"

        exc = np.einsum(
            einstr,
            np.conj(mvc),
            vq0g_local,
            mvcp,
            optimize=True,
        )

        return exc #* self.kernel_factor
    
    
    def calc_kernel_mtxel(
            self,
            parallel:bool = True,
    ):
        numk = self.kpts.numk

        num_val = self.charge_mtxel.val_num
        num_con = self.charge_mtxel.con_num

        # Store the value of v(q + G) for q = 0.
        vq0g = np.array(self.vcoul.vcoul[0])

        # Sort the indices of v(q + G), for q = 0.
        sort_order_0 = sort_cryst_like_BGW(
            self.l_gq[0].gk_cryst, self.l_gq[0].gk_norm2
        )        
        vq0g = vq0g[sort_order_0]

        if self.in_parallel and parallel:
            proc_rank = self.comm.Get_rank()

            if proc_rank == 0:
                k_indices, kp_indices = np.meshgrid(
                    np.arange(numk),
                    np.arange(numk),
                    indexing="ij",
                )
                full_pairs = np.stack(
                    (k_indices.flatten(), kp_indices.flatten()), axis=1
                )
                chunks = np.array_split(full_pairs, self.comm.Get_size())
            else:
                chunks = None

            local_pairs = self.comm.scatter(chunks, root=0)

            # Each element: (k_idx, kp_idx, exc_block, head_block, wings_block, body_block)
            local_blocks = []  

            for pair in local_pairs:
                k_idx, kp_idx = pair

                mvc = self.charge_mtxel.mvc(k_idx)

                mvcp = self.charge_mtxel.mvc(kp_idx)

                data_mvvp = self.charge_mtxel.mvvp(
                    k_idx = k_idx,
                    kp_idx = kp_idx,
                    ret_q = 'True'
                )

                mvvp = data_mvvp[0]
                q_idx = data_mvvp[1]

                mccp = self.charge_mtxel.mccp(
                    k_idx = k_idx,
                    kp_idx = kp_idx,
                )

                vqg = self.vcoul.vcoul[q_idx]
                vqg = np.array(vqg)

                # Sort the indices of v(q + G).
                sort_order = sort_cryst_like_BGW(
                    self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2
                )
                vqg = vqg[sort_order]

                # Indices of epsinv were already sorted.
                epsinv = self.l_epsinv[q_idx]

                # Calculate the head, wings, body and exchange part of the kernel.
                head = self.calc_head(
                    q_idx = q_idx,
                    vqg = vqg,
                    epsinv = epsinv,
                    mccp = mccp,
                    mvvp = mvvp,
                )

                wings = self.calc_wings(
                    q_idx = q_idx,
                    vqg = vqg,
                    epsinv = epsinv,
                    mccp = mccp,
                    mvvp = mvvp,
                )

                body = self.calc_body(
                    vqg = vqg,
                    epsinv = epsinv,
                    mccp = mccp,
                    mvvp = mvvp,
                )

                exc = self.calc_exc(
                    vq0g = vq0g,
                    mvc = mvc,
                    mvcp = mvcp,
                )

                local_blocks.append(
                    (k_idx, kp_idx, exc, head, wings, body)
                )
            
            if proc_rank != 0:
                self.comm.send(local_blocks, dest=0, tag=77)

                exc_result = self.comm.bcast(None, root = 0)
                head_result = self.comm.bcast(None, root = 0)

                wings_result = self.comm.bcast(None, root = 0)
                body_result = self.comm.bcast(None, root = 0)

            else:
                exc_mtx = np.zeros(
                    (numk, numk, num_val, num_val, num_con, num_con),
                    dtype = complex
                )

                head_mtx = np.zeros(
                    (numk, numk, num_val, num_val, num_con, num_con),
                    dtype = complex
                )

                wings_mtx = np.zeros(
                    (numk, numk, num_val, num_val, num_con, num_con),
                    dtype = complex
                )

                body_mtx = np.zeros(
                    (numk, numk, num_val, num_val, num_con, num_con),
                    dtype = complex
                )

                for k_idx, kp_idx, exc, head, wings, body in local_blocks:
                    exc_mtx[k_idx, kp_idx] += exc
                    head_mtx[k_idx, kp_idx] += head

                    wings_mtx[k_idx, kp_idx] += wings
                    body_mtx[k_idx, kp_idx] += body

                for source in range(1, self.comm_size):
                    remote_blocks = self.comm.recv(source=source, tag=77)

                    for k_idx, kp_idx, exc, head, wings, body in remote_blocks:
                        exc_mtx[k_idx, kp_idx] += exc
                        head_mtx[k_idx, kp_idx] += head

                        wings_mtx[k_idx, kp_idx] += wings
                        body_mtx[k_idx, kp_idx] += body

                exc_result = exc_mtx
                head_result = head_mtx

                wings_result = wings_mtx
                body_result = body_mtx

                # Broadcast the results to all processes.
                self.comm.bcast(exc_result, root=0)
                self.comm.bcast(head_result, root=0)

                self.comm.bcast(wings_result, root=0)
                self.comm.bcast(body_result, root=0)

            return exc_result, head_result, wings_result, body_result

        else:
            exc_result = np.zeros(
                (numk, numk, num_val, num_val, num_con, num_con),
                dtype=complex,
            )

            head_result = np.zeros(
                (numk, numk, num_val, num_val, num_con, num_con),
                dtype=complex,
            )

            wings_result = np.zeros(
                (numk, numk, num_val, num_val, num_con, num_con),
                dtype=complex,
            )

            body_result = np.zeros(
                (numk, numk, num_val, num_val, num_con, num_con),
                dtype=complex,
            )

            for k_idx in range(numk):
                for kp_idx in range(numk):

                    mvc = self.charge_mtxel.mvc(k_idx)
                    mvcp = self.charge_mtxel.mvc(kp_idx)

                    data_mvvp = self.charge_mtxel.mvvp(
                        k_idx=k_idx,
                        kp_idx=kp_idx,
                        ret_q='True'
                    )

                    mvvp = data_mvvp[0]
                    q_idx = data_mvvp[1]

                    print(q_idx)

                    mccp = self.charge_mtxel.mccp(
                        k_idx=k_idx,
                        kp_idx=kp_idx,
                    )

                    vqg = self.vcoul.vcoul[q_idx]
                    vqg = np.array(vqg)

                    # Sort the indices of v(q + G).
                    sort_order = sort_cryst_like_BGW(
                        self.l_gq[q_idx].gk_cryst, self.l_gq[q_idx].gk_norm2
                    )
                    vqg = vqg[sort_order]

                    # Indices of epsinv were already sorted.
                    epsinv = self.l_epsinv[q_idx]

                    # Calculate the head, wings and body of the kernel.
                    head = self.calc_head(
                        q_idx=q_idx,
                        vqg=vqg,
                        epsinv=epsinv,
                        mccp=mccp,
                        mvvp=mvvp,
                    )

                    wings = self.calc_wings(
                        q_idx=q_idx,
                        vqg=vqg,
                        epsinv=epsinv,
                        mccp=mccp,
                        mvvp=mvvp,
                    )

                    body = self.calc_body(
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

                    exc_result[k_idx, kp_idx] += exc
                    head_result[k_idx, kp_idx] += head

                    wings_result[k_idx, kp_idx] += wings
                    body_result[k_idx, kp_idx] += body

            return exc_result, head_result, wings_result, body_result
        










