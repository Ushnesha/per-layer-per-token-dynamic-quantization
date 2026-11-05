"""Named vLLM worker RPC methods; only method names cross process boundaries."""

WORKER_EXTENSION = 'src.evaluation.benchmark.worker_extension.EvaluationWorkerExtension'


class EvaluationWorkerExtension:
    def aditya_install_memory_tracking(self):
        from .runtime_memory import install_memory_tracking
        return install_memory_tracking(self)

    def aditya_remove_memory_tracking(self):
        from .runtime_memory import remove_memory_tracking
        return remove_memory_tracking(self)

    def aditya_install_timing(self):
        from .kernel_time import install_timing
        return install_timing(self)

    def aditya_remove_timing(self):
        from .kernel_time import remove_timing
        return remove_timing(self)

    def aditya_install_ranges(self):
        from .profiling_workload import install_ranges
        return install_ranges(self)

    def aditya_remove_ranges(self):
        from .profiling_workload import remove_ranges
        return remove_ranges(self)

    def aditya_collect_components(self):
        from .component_memory import collect_worker_components
        return collect_worker_components(self)
