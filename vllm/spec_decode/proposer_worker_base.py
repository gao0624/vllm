from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.interfaces import SpeculativeProposer, MultiLevelSpeculativeProposals, SpeculativeProposals
from vllm.worker.worker_base import LoraNotSupportedWorkerBase


class ProposerWorkerBase(LoraNotSupportedWorkerBase, SpeculativeProposer):
    """Interface for proposer workers"""

    @abstractmethod
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        # A set containing all sequence IDs that were assigned bonus tokens
        # in their last forward pass. This set is used to backfill the KV cache
        # with the key-value pairs of the penultimate token in the sequences.
        # This parameter is only used by the MultiStepWorker, which relies on
        # the KV cache for token generation. It is not used by workers that
        # do not utilize the KV cache.
        seq_ids_with_bonus_token_in_last_step: Set[int]
    ) -> Tuple[Optional[List[SamplerOutput]], bool]:
        raise NotImplementedError

    def set_include_gpu_probs_tensor(self) -> None:
        """Implementation optional"""
        pass

    def set_should_modify_greedy_probs_inplace(self) -> None:
        """Implementation optional"""
        pass

    # 新增多级proposal默认实现(可在子类重写)
    # def get_multi_level_spec_proposals(
    #     self,
    #     execute_model_req: ExecuteModelRequest,
    #     # If set, this contains all sequence IDs that were assigned
    #     # bonus tokens in their last forward pass.
    #     seq_ids_with_bonus_token_in_last_step: Set[int],
    #     num_levels: int,
    # ) -> List[SamplerOutput]:
    #     """
    #     默认多级 proposal 实现：循环调用 get_spec_proposals。
    #     子类可按需重载以自定义多级推测策略。
    #     """
    #     proposals_list = []
    #     curr_execute_model_req = execute_model_req
    #     curr_seq_ids = seq_ids_with_bonus_token_in_last_step
    #     for _ in range(num_levels):
    #         # 获取当前级别的 proposal
    #         proposal = self.get_spec_proposals(
    #             curr_execute_model_req, curr_seq_ids
    #         )
    #         proposals_list.append(proposal)
    #         curr_execute_model_req = self._append_proposal_to_execute_model_req(curr_execute_model_req, proposal)
    #         return MultiLevelSpeculativeProposals(multi_level_proposals=proposals_list)

    # # 这里给个空实现，实际功能应在具体 worker 类里完善
    # def _append_proposal_to_execute_model_req(self, execute_model_req, proposal):
    #     # 直接返回原始请求，不做递进（基类只是占位，具体逻辑子类实现）
    #     return execute_model_req

class NonLLMProposerWorkerBase(ProposerWorkerBase, ABC):
    """Proposer worker which does not use a model with kvcache"""

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """get_spec_proposals is used to get the proposals"""
        return []

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """This is never called on the proposer, only the target model"""
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        return 0
