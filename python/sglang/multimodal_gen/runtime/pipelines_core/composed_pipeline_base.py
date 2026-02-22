# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import torch
from tqdm import tqdm

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ComposedPipelineBase(ABC):
    """
    Base class for pipelines composed of multiple stages.

    This class provides the framework for creating pipelines by composing multiple
    stages together. Each stage is responsible for a specific part of the diffusion
    process, and the pipeline orchestrates the execution of these stages.
    """

    is_video_pipeline: bool = False  # To be overridden by video pipelines
    # should contains only the modules to be loaded
    _required_config_modules: list[str] = []
    _extra_config_module_map: dict[str, str] = {}
    server_args: ServerArgs | None = None
    modules: dict[str, Any] = {}
    executor: PipelineExecutor | None = None

    # the name of the pipeline it associated with, in diffusers
    pipeline_name: str

    def is_lora_effective(self):
        return False

    def is_lora_set(self):
        return False

    def __init__(
        self,
        model_path: str,
        server_args: ServerArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        executor: PipelineExecutor | None = None,
    ):
        """
        Initialize the pipeline. After __init__, the pipeline should be ready to
        use. The pipeline should be stateless and not hold any batch state.
        """
        self.server_args = server_args

        self.model_path: str = model_path
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}
        self.executor = executor or self.build_executor(server_args=server_args)

        if required_config_modules is not None:
            self._required_config_modules = required_config_modules

        if self._required_config_modules is None:
            raise NotImplementedError("Subclass must set _required_config_modules")

        # [module_name, gpu memory usage]
        self.memory_usages: dict[str, float] = {}
        # [module_name, (start_time, end_time)] relative to load start
        self.load_timings: dict[str, tuple[float, float]] = {}
        # [module_name, [(phase_name, start, end), ...]] relative to load start
        self.phase_timings: dict[str, list[tuple[str, float, float]]] = {}
        # Load modules directly in initialization
        logger.info("Loading pipeline modules...")
        self.modules = self.load_modules(server_args, loaded_modules)

        self.__post_init__()

    def build_executor(self, server_args: ServerArgs):
        # TODO
        from sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor import (
            ParallelExecutor,
        )

        # return SyncExecutor(server_args=server_args)
        return ParallelExecutor(server_args=server_args)

    def __post_init__(self) -> None:
        assert self.server_args is not None, "server_args must be set"
        self.initialize_pipeline(self.server_args)

        logger.info("Creating pipeline stages...")
        self.create_pipeline_stages(self.server_args)

    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        if module_name not in self.modules:
            return default_value
        return self.modules[module_name]

    def add_module(self, module_name: str, module: Any):
        self.modules[module_name] = module

    def _load_config(self) -> dict[str, Any]:
        model_path = maybe_download_model(self.model_path)
        self.model_path = model_path
        # server_args.downloaded_model_path = model_path
        logger.info("Model path: %s", model_path)
        config = verify_model_config_and_directory(model_path)
        return cast(dict[str, Any], config)

    @property
    def required_config_modules(self) -> list[str]:
        """
        List of modules that are required by the pipeline. The names should match
        the diffusers directory and model_index.json file. These modules will be
        loaded using the PipelineComponentLoader and made available in the
        modules dictionary. Access these modules using the get_module method.

        class ConcretePipeline(ComposedPipelineBase):
            _required_config_modules = ["vae", "text_encoder", "transformer", "scheduler", "tokenizer"]


            @property
            def required_config_modules(self):
                return self._required_config_modules
        """
        return self._required_config_modules

    @property
    def stages(self) -> list[PipelineStage]:
        """
        List of stages in the pipeline.
        """
        return self._stages

    @abstractmethod
    def create_pipeline_stages(self, server_args: ServerArgs):
        """
        Create the inference pipeline stages.
        """
        raise NotImplementedError

    def initialize_pipeline(self, server_args: ServerArgs):
        """
        Initialize the pipeline.
        """
        return

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Load the modules from the config.
        loaded_modules: Optional[Dict[str, torch.nn.Module]] = None,
        If provided, loaded_modules will be used instead of loading from config/pretrained weights.
        """

        model_index = self._load_config()
        logger.info("Loading pipeline modules from config: %s", model_index)

        # remove keys that are not pipeline modules
        model_index.pop("_class_name")
        model_index.pop("_diffusers_version")
        if (
            "boundary_ratio" in model_index
            and model_index["boundary_ratio"] is not None
        ):
            has_transformer = (
                "transformer" in model_index
                or "transformer_2" in model_index
                or "transformer" in self.required_config_modules
                or "transformer_2" in self.required_config_modules
            )
            if has_transformer:
                logger.info(
                    "MoE pipeline detected. Adding transformer_2 to self.required_config_modules..."
                )
                if "transformer_2" not in self.required_config_modules:
                    self.required_config_modules.append("transformer_2")
            else:
                logger.info(
                    "Boundary ratio found in model_index.json without transformers; "
                    "using it for pipeline config only."
                )
            logger.info(
                "Setting boundary ratio to %s",
                model_index["boundary_ratio"],
            )
            server_args.pipeline_config.dit_config.boundary_ratio = model_index[
                "boundary_ratio"
            ]

        model_index.pop("boundary_ratio", None)
        # used by Wan2.2 ti2v
        model_index.pop("expand_timesteps", None)

        # some sanity checks
        assert (
            len(model_index) > 1
        ), "model_index.json must contain at least one pipeline module"

        model_index = {
            required_module: model_index[required_module]
            for required_module in self.required_config_modules
        }

        for module_name in self.required_config_modules:
            if (
                module_name not in model_index
                and module_name in self._extra_config_module_map
            ):
                extra_module_value = self._extra_config_module_map[module_name]
                logger.warning(
                    "model_index.json does not contain a %s module, but found {%s: %s} in _extra_config_module_map, adding to model_index.",
                    module_name,
                    module_name,
                    extra_module_value,
                )
                if extra_module_value in model_index:
                    logger.info(
                        "Using module %s for %s", extra_module_value, module_name
                    )
                    model_index[module_name] = model_index[extra_module_value]
                    continue
                else:
                    raise ValueError(
                        f"Required module key: {module_name} value: {model_index.get(module_name)} was not found in loaded modules {model_index.keys()}"
                    )

        # all the component models used by the pipeline
        required_modules = self.required_config_modules
        logger.info("Loading required components: %s", required_modules)

        loaded_components = {}

        # Pre-pass: resolve paths and handle null entries / already-provided modules.
        # This part mutates self (e.g. removing null modules from required list) so
        # it must run sequentially before any parallel work.
        to_load: list[
            tuple[str, str, str, str]
        ] = []  # (module_name, load_module_name, component_model_path, transformers_or_diffusers)
        for module_name, (
            transformers_or_diffusers,
            architecture,
        ) in model_index.items():
            if transformers_or_diffusers is None:
                logger.warning(
                    "Module %s in model_index.json has null value, removing from required_config_modules",
                    module_name,
                )
                if module_name in self.required_config_modules:
                    self.required_config_modules.remove(module_name)
                continue
            if module_name not in required_modules:
                logger.info("Skipping module %s", module_name)
                continue
            if loaded_modules is not None and module_name in loaded_modules:
                logger.info("Using module %s already provided", module_name)
                loaded_components[module_name] = loaded_modules[module_name]
                continue

            # Resolve the load name (may differ via extra config module map)
            if module_name in self._extra_config_module_map:
                load_module_name = self._extra_config_module_map[module_name]
            else:
                load_module_name = module_name

            # Resolve model path
            if module_name == "vae" and server_args.vae_path is not None:
                component_model_path = server_args.vae_path
                if not os.path.exists(component_model_path):
                    component_model_path = maybe_download_model(component_model_path)
                logger.info(
                    "Using custom VAE path: %s instead of default path: %s",
                    component_model_path,
                    os.path.join(self.model_path, load_module_name),
                )
            else:
                component_model_path = os.path.join(self.model_path, load_module_name)

            to_load.append(
                (
                    module_name,
                    load_module_name,
                    component_model_path,
                    transformers_or_diffusers,
                )
            )

        load_t0 = time.perf_counter()

        def _load_one(
            module_name: str,
            load_module_name: str,
            component_model_path: str,
            transformers_or_diffusers: str,
        ):
            """Load a single component. Returns (module_name, load_module_name, module, memory_usage, t_start, t_end, phase_timings)."""
            t_start = time.perf_counter() - load_t0
            module, memory_usage, raw_phases = PipelineComponentLoader.load_component(
                component_name=load_module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                server_args=server_args,
            )
            t_end = time.perf_counter() - load_t0
            # Re-base absolute phase timestamps relative to load_t0
            phases = [
                (name, abs_start - load_t0, abs_end - load_t0)
                for name, abs_start, abs_end in raw_phases
            ]
            return module_name, load_module_name, module, memory_usage, t_start, t_end, phases

        use_parallel = getattr(server_args, "parallel_loading", False) and len(to_load) > 1

        if use_parallel:
            logger.info(
                "Loading %d modules in parallel (threads=%d)",
                len(to_load),
                len(to_load),
            )
            with ThreadPoolExecutor(max_workers=len(to_load)) as pool:
                futures = {
                    pool.submit(_load_one, *args): args[0] for args in to_load
                }
                # Use tqdm to show progress
                with tqdm(
                    total=len(futures),
                    desc="Loading required modules (parallel)",
                ) as pbar:
                    for future in as_completed(futures):
                        module_name, load_module_name, module, memory_usage, t_start, t_end, phases = (
                            future.result()
                        )
                        self.memory_usages[load_module_name] = memory_usage
                        self.load_timings[load_module_name] = (t_start, t_end)
                        self.phase_timings[load_module_name] = phases
                        if module_name in loaded_components:
                            logger.warning("Overwriting module %s", module_name)
                        loaded_components[module_name] = module
                        pbar.update(1)
        else:
            # Sequential loading (original path)
            for (
                module_name,
                load_module_name,
                component_model_path,
                transformers_or_diffusers,
            ) in tqdm(iterable=to_load, desc="Loading required modules"):
                module_name, load_module_name, module, memory_usage, t_start, t_end, phases = _load_one(
                    module_name,
                    load_module_name,
                    component_model_path,
                    transformers_or_diffusers,
                )
                self.memory_usages[load_module_name] = memory_usage
                self.load_timings[load_module_name] = (t_start, t_end)
                self.phase_timings[load_module_name] = phases
                if module_name in loaded_components:
                    logger.warning("Overwriting module %s", module_name)
                loaded_components[module_name] = module

        # Post-pass: validate all required modules were loaded
        for module_name in required_modules:
            if (
                module_name not in loaded_components
                or loaded_components[module_name] is None
            ):
                raise ValueError(
                    f"Required module: {module_name} was not found in loaded modules: {list(loaded_components.keys())}"
                )

        logger.debug(
            "Memory usage of loaded modules (GiB): %s. Available memory: %s",
            self.memory_usages,
            round(current_platform.get_available_gpu_memory(), 2),
        )

        return loaded_components

    def add_stage(self, stage_name: str, stage: PipelineStage):
        assert self.modules is not None, "No modules are registered"
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)

    # TODO(will): don't hardcode no_grad
    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Generate a video or image using the pipeline.

        Args:
            batch: The batch to generate from.
            server_args: The inference arguments.
        Returns:
            Req: The batch with the generated video or image.
        """

        if self.is_lora_set() and not self.is_lora_effective():
            logger.warning(
                "LoRA adapter is set, but not effective. Please make sure the LoRA weights are merged"
            )

        # Execute each stage
        if not batch.is_warmup and not batch.suppress_logs:
            logger.info(
                "Running pipeline stages: %s",
                list(self._stage_name_mapping.keys()),
                main_process_only=True,
            )

        return self.executor.execute_with_profiling(self.stages, batch, server_args)
