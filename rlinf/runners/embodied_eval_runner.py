# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import (
    compute_evaluate_metrics,
    compute_grouped_success_metrics,
)

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        # Per-task / per-suite breakdown (LIBERO only): the env tags each eval
        # trajectory with its task_id, which we group here using a static
        # task_id -> suite map derived from the eval suite name.
        grouped = compute_grouped_success_metrics(
            eval_metrics_list, task_to_suite=self._get_task_to_suite()
        )
        eval_metrics.update(grouped)
        return eval_metrics

    def _get_task_to_suite(self):
        """Build (and cache) the task_id -> suite map for the eval benchmark.

        Returns None for non-LIBERO envs, in which case only the overall /
        per-task breakdown (if any task_id is present) is reported.
        """
        if getattr(self, "_task_to_suite", "unset") != "unset":
            return self._task_to_suite

        self._task_to_suite = None
        try:
            eval_cfg = self.cfg.env.eval
            if str(eval_cfg.get("env_type", "")).lower() == "libero":
                from rlinf.envs.libero.utils import get_task_suite_map

                self._task_to_suite = get_task_suite_map(eval_cfg.task_suite_name)
        except Exception as e:  # noqa: BLE001 - reporting must never break eval
            self.logger.warning(f"Could not build task->suite map: {e}")
        return self._task_to_suite

    def run(self):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_logger.finish()
