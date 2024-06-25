"""This module contains the evaluator classes for evaluating runs."""

import asyncio
import inspect
import uuid
from abc import abstractmethod
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

from typing_extensions import TypedDict

try:
    from pydantic.v1 import BaseModel, Field, ValidationError  # type: ignore[import]
except ImportError:
    from pydantic import BaseModel, Field, ValidationError  # type: ignore[assignment]

from functools import wraps

from langsmith.schemas import SCORE_TYPE, VALUE_TYPE, Example, Run


class Category(TypedDict):
    """A category for categorical feedback."""

    value: Optional[Union[float, int]]
    """The numeric score/ordinal corresponding to this category."""
    label: str
    """The label for this category."""


class FeedbackConfig(TypedDict, total=False):
    """Configuration to define a type of feedback.

    Applied on on the first creation of a feedback_key.
    """

    type: Literal["continuous", "categorical", "freeform"]
    """The type of feedback."""
    min: Optional[Union[float, int]]
    """The minimum permitted value (if continuous type)."""
    max: Optional[Union[float, int]]
    """The maximum value permitted value (if continuous type)."""
    categories: Optional[List[Union[Category, dict]]]


class EvaluationResult(BaseModel):
    """Evaluation result."""

    key: str
    """The aspect, metric name, or label for this evaluation."""
    score: SCORE_TYPE = None
    """The numeric score for this evaluation."""
    value: VALUE_TYPE = None
    """The value for this evaluation, if not numeric."""
    comment: Optional[str] = None
    """An explanation regarding the evaluation."""
    correction: Optional[Dict] = None
    """What the correct value should be, if applicable."""
    evaluator_info: Dict = Field(default_factory=dict)
    """Additional information about the evaluator."""
    feedback_config: Optional[Union[FeedbackConfig, dict]] = None
    """The configuration used to generate this feedback."""
    source_run_id: Optional[Union[uuid.UUID, str]] = None
    """The ID of the trace of the evaluator itself."""
    target_run_id: Optional[Union[uuid.UUID, str]] = None
    """The ID of the trace this evaluation is applied to.
    
    If none provided, the evaluation feedback is applied to the
    root trace being."""

    class Config:
        """Pydantic model configuration."""

        allow_extra = False


class EvaluationResults(TypedDict, total=False):
    """Batch evaluation results.

    This makes it easy for your evaluator to return multiple
    metrics at once.
    """

    results: List[EvaluationResult]
    """The evaluation results."""


class RunEvaluator:
    """Evaluator interface class."""

    @abstractmethod
    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> Union[EvaluationResult, EvaluationResults]:
        """Evaluate an example."""

    async def aevaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> Union[EvaluationResult, EvaluationResults]:
        """Evaluate an example asynchronously."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.evaluate_run, run, example
        )


_RUNNABLE_OUTPUT = Union[EvaluationResult, EvaluationResults, dict]


class ComparisonEvaluationResult(BaseModel):
    """Feedback scores for the results of comparative evaluations.

    These are generated by functions that compare two or more runs,
    returning a ranking or other feedback.
    """

    key: str
    """The aspect, metric name, or label for this evaluation."""
    scores: Dict[Union[uuid.UUID, str], SCORE_TYPE]
    """The scores for each run in the comparison."""
    source_run_id: Optional[Union[uuid.UUID, str]] = None
    """The ID of the trace of the evaluator itself."""


_COMPARISON_OUTPUT = Union[ComparisonEvaluationResult, dict]


class DynamicRunEvaluator(RunEvaluator):
    """A dynamic evaluator that wraps a function and transforms it into a `RunEvaluator`.

    This class is designed to be used with the `@run_evaluator` decorator, allowing
    functions that take a `Run` and an optional `Example` as arguments, and return
    an `EvaluationResult` or `EvaluationResults`, to be used as instances of `RunEvaluator`.

    Attributes:
        func (Callable): The function that is wrapped by this evaluator.
    """  # noqa: E501

    def __init__(
        self,
        func: Callable[
            [Run, Optional[Example]],
            Union[_RUNNABLE_OUTPUT, Awaitable[_RUNNABLE_OUTPUT]],
        ],
        # Async function to be used for async evaluation. Optional
        afunc: Optional[
            Callable[
                [Run, Optional[Example]],
                Awaitable[_RUNNABLE_OUTPUT],
            ]
        ] = None,
    ):
        """Initialize the DynamicRunEvaluator with a given function.

        Args:
            func (Callable): A function that takes a `Run` and an optional `Example` as
            arguments, and returns a dict or `ComparisonEvaluationResult`.
        """
        wraps(func)(self)
        from langsmith import run_helpers  # type: ignore

        if afunc is not None:
            self.afunc = run_helpers.ensure_traceable(afunc)
            self._name = getattr(afunc, "__name__", "DynamicRunEvaluator")
        if inspect.iscoroutinefunction(func):
            if afunc is not None:
                raise TypeError(
                    "Func was provided as a coroutine function, but afunc was "
                    "also provided. If providing both, func should be a regular "
                    "function to avoid ambiguity."
                )
            self.afunc = run_helpers.ensure_traceable(func)
            self._name = getattr(func, "__name__", "DynamicRunEvaluator")
        else:
            self.func = run_helpers.ensure_traceable(
                cast(Callable[[Run, Optional[Example]], _RUNNABLE_OUTPUT], func)
            )
            self._name = getattr(func, "__name__", "DynamicRunEvaluator")

    def _coerce_evaluation_result(
        self,
        result: Union[EvaluationResult, dict],
        source_run_id: uuid.UUID,
        allow_no_key: bool = False,
    ) -> EvaluationResult:
        if isinstance(result, EvaluationResult):
            if not result.source_run_id:
                result.source_run_id = source_run_id
            return result
        try:
            if "key" not in result:
                if allow_no_key:
                    result["key"] = self._name
            return EvaluationResult(**{"source_run_id": source_run_id, **result})
        except ValidationError as e:
            raise ValueError(
                "Expected an EvaluationResult object, or dict with a metric"
                f" 'key' and optional 'score'; got {result}"
            ) from e

    def _coerce_evaluation_results(
        self,
        results: Union[dict, EvaluationResults],
        source_run_id: uuid.UUID,
    ) -> Union[EvaluationResult, EvaluationResults]:
        if "results" in results:
            cp = results.copy()
            cp["results"] = [
                self._coerce_evaluation_result(r, source_run_id=source_run_id)
                for r in results["results"]
            ]
            return EvaluationResults(**cp)

        return self._coerce_evaluation_result(
            cast(dict, results), allow_no_key=True, source_run_id=source_run_id
        )

    def _format_result(
        self,
        result: Union[EvaluationResult, EvaluationResults, dict],
        source_run_id: uuid.UUID,
    ) -> Union[EvaluationResult, EvaluationResults]:
        if isinstance(result, EvaluationResult):
            if not result.source_run_id:
                result.source_run_id = source_run_id
            return result
        if not isinstance(result, dict):
            raise ValueError(
                f"Expected a dict, EvaluationResult, or EvaluationResults, got {result}"
            )
        return self._coerce_evaluation_results(result, source_run_id)

    @property
    def is_async(self) -> bool:
        """Check if the evaluator function is asynchronous.

        Returns:
            bool: True if the evaluator function is asynchronous, False otherwise.
        """
        return hasattr(self, "afunc")

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> Union[EvaluationResult, EvaluationResults]:
        """Evaluate a run using the wrapped function.

        This method directly invokes the wrapped function with the provided arguments.

        Args:
            run (Run): The run to be evaluated.
            example (Optional[Example]): An optional example to be used in the evaluation.

        Returns:
            Union[EvaluationResult, EvaluationResults]: The result of the evaluation.
        """  # noqa: E501
        if not hasattr(self, "func"):
            running_loop = asyncio.get_event_loop()
            if running_loop.is_running():
                raise RuntimeError(
                    "Cannot call `evaluate_run` on an async run evaluator from"
                    " within an running event loop. Use `aevaluate_run` instead."
                )
            else:
                return running_loop.run_until_complete(self.aevaluate_run(run, example))
        source_run_id = uuid.uuid4()
        metadata: Dict[str, Any] = {"target_run_id": run.id}
        if getattr(run, "session_id", None):
            metadata["experiment"] = str(run.session_id)
        result = self.func(
            run,
            example,
            langsmith_extra={"run_id": source_run_id, "metadata": metadata},
        )
        return self._format_result(result, source_run_id)

    async def aevaluate_run(self, run: Run, example: Optional[Example] = None):
        """Evaluate a run asynchronously using the wrapped async function.

        This method directly invokes the wrapped async function with the
            provided arguments.

        Args:
            run (Run): The run to be evaluated.
            example (Optional[Example]): An optional example to be used
                in the evaluation.

        Returns:
            Union[EvaluationResult, EvaluationResults]: The result of the evaluation.
        """
        if not hasattr(self, "afunc"):
            return await super().aevaluate_run(run, example)
        source_run_id = uuid.uuid4()
        metadata: Dict[str, Any] = {"target_run_id": run.id}
        if getattr(run, "session_id", None):
            metadata["experiment"] = str(run.session_id)
        result = await self.afunc(
            run,
            example,
            langsmith_extra={"run_id": source_run_id, "metadata": metadata},
        )
        return self._format_result(result, source_run_id)

    def __call__(
        self, run: Run, example: Optional[Example] = None
    ) -> Union[EvaluationResult, EvaluationResults]:
        """Make the evaluator callable, allowing it to be used like a function.

        This method enables the evaluator instance to be called directly, forwarding the
        call to `evaluate_run`.

        Args:
            run (Run): The run to be evaluated.
            example (Optional[Example]): An optional example to be used in the evaluation.

        Returns:
            Union[EvaluationResult, EvaluationResults]: The result of the evaluation.
        """  # noqa: E501
        return self.evaluate_run(run, example)

    def __repr__(self) -> str:
        """Represent the DynamicRunEvaluator object."""
        return f"<DynamicRunEvaluator {getattr(self.func, '__name__')}>"


def run_evaluator(
    func: Callable[
        [Run, Optional[Example]], Union[_RUNNABLE_OUTPUT, Awaitable[_RUNNABLE_OUTPUT]]
    ],
):
    """Create a run evaluator from a function.

    Decorator that transforms a function into a `RunEvaluator`.
    """
    return DynamicRunEvaluator(func)


class DynamicComparisonRunEvaluator:
    """Compare predictions (as traces) from 2 or more runs."""

    def __init__(
        self,
        func: Callable[
            [Sequence[Run], Optional[Example]],
            Union[_COMPARISON_OUTPUT, Awaitable[_COMPARISON_OUTPUT]],
        ],
        # Async function to be used for async evaluation. Optional
        afunc: Optional[
            Callable[
                [Sequence[Run], Optional[Example]],
                Awaitable[_COMPARISON_OUTPUT],
            ]
        ] = None,
    ):
        """Initialize the DynamicRunEvaluator with a given function.

        Args:
            func (Callable): A function that takes a `Run` and an optional `Example` as
            arguments, and returns an `EvaluationResult` or `EvaluationResults`.
        """
        wraps(func)(self)
        from langsmith import run_helpers  # type: ignore

        if afunc is not None:
            self.afunc = run_helpers.ensure_traceable(afunc)
            self._name = getattr(afunc, "__name__", "DynamicRunEvaluator")
        if inspect.iscoroutinefunction(func):
            if afunc is not None:
                raise TypeError(
                    "Func was provided as a coroutine function, but afunc was "
                    "also provided. If providing both, func should be a regular "
                    "function to avoid ambiguity."
                )
            self.afunc = run_helpers.ensure_traceable(func)
            self._name = getattr(func, "__name__", "DynamicRunEvaluator")
        else:
            self.func = run_helpers.ensure_traceable(
                cast(
                    Callable[
                        [Sequence[Run], Optional[Example]],
                        _COMPARISON_OUTPUT,
                    ],
                    func,
                )
            )
            self._name = getattr(func, "__name__", "DynamicRunEvaluator")

    @property
    def is_async(self) -> bool:
        """Check if the evaluator function is asynchronous.

        Returns:
            bool: True if the evaluator function is asynchronous, False otherwise.
        """
        return hasattr(self, "afunc")

    def compare_runs(
        self, runs: Sequence[Run], example: Optional[Example] = None
    ) -> ComparisonEvaluationResult:
        """Compare runs to score preferences.

        Args:
            runs: A list of runs to compare.
            example: An optional example to be used in the evaluation.

        """  # noqa: E501
        if not hasattr(self, "func"):
            running_loop = asyncio.get_event_loop()
            if running_loop.is_running():
                raise RuntimeError(
                    "Cannot call `evaluate_run` on an async run evaluator from"
                    " within an running event loop. Use `aevaluate_run` instead."
                )
            else:
                return running_loop.run_until_complete(
                    self.acompare_runs(runs, example)
                )
        source_run_id = uuid.uuid4()
        tags = self._get_tags(runs)
        # TODO: Add metadata for the "comparison experiment" here
        result = self.func(
            runs,
            example,
            langsmith_extra={"run_id": source_run_id, "tags": tags},
        )
        return self._format_results(result, source_run_id)

    async def acompare_runs(
        self, runs: Sequence[Run], example: Optional[Example] = None
    ) -> ComparisonEvaluationResult:
        """Evaluate a run asynchronously using the wrapped async function.

        This method directly invokes the wrapped async function with the
            provided arguments.

        Args:
            run (Run): The run to be evaluated.
            example (Optional[Example]): An optional example to be used
                in the evaluation.

        Returns:
            ComparisonEvaluationResult: The result of the evaluation.
        """
        if not hasattr(self, "afunc"):
            return self.compare_runs(runs, example)
        source_run_id = uuid.uuid4()
        tags = self._get_tags(runs)
        # TODO: Add metadata for the "comparison experiment" here
        result = await self.afunc(
            runs,
            example,
            langsmith_extra={"run_id": source_run_id, "tags": tags},
        )
        return self._format_results(result, source_run_id)

    def __call__(
        self, runs: Sequence[Run], example: Optional[Example] = None
    ) -> ComparisonEvaluationResult:
        """Make the evaluator callable, allowing it to be used like a function.

        This method enables the evaluator instance to be called directly, forwarding the
        call to `evaluate_run`.

        Args:
            run (Run): The run to be evaluated.
            example (Optional[Example]): An optional example to be used in the evaluation.

        Returns:
            ComparisonEvaluationResult: The result of the evaluation.
        """  # noqa: E501
        return self.compare_runs(runs, example)

    def __repr__(self) -> str:
        """Represent the DynamicRunEvaluator object."""
        return f"<DynamicComparisonRunEvaluator {getattr(self.func, '__name__')}>"

    @staticmethod
    def _get_tags(runs: Sequence[Run]) -> List[str]:
        """Extract tags from runs."""
        # Add tags to support filtering
        tags = []
        for run in runs:
            tags.append("run:" + str(run.id))
            if getattr(run, "session_id", None):
                tags.append("experiment:" + str(run.session_id))
        return tags

    def _coerce_evaluation_result(
        self,
        result: Union[EvaluationResult, dict],
        source_run_id: uuid.UUID,
        allow_no_key: bool = False,
    ) -> EvaluationResult:
        if isinstance(result, EvaluationResult):
            if not result.source_run_id:
                result.source_run_id = source_run_id
            return result
        try:
            if "key" not in result:
                if allow_no_key:
                    result["key"] = self._name
            return EvaluationResult(**{"source_run_id": source_run_id, **result})
        except ValidationError as e:
            raise ValueError(
                "Expected an EvaluationResult object, or dict with a metric"
                f" 'key' and optional 'score'; got {result}"
            ) from e

    def _coerce_evaluation_results(
        self,
        results: Union[dict, EvaluationResults],
        source_run_id: uuid.UUID,
    ) -> Union[EvaluationResult, EvaluationResults]:
        if "results" in results:
            cp = results.copy()
            cp["results"] = [
                self._coerce_evaluation_result(r, source_run_id=source_run_id)
                for r in results["results"]
            ]
            return EvaluationResults(**cp)

        return self._coerce_evaluation_result(
            cast(dict, results), allow_no_key=True, source_run_id=source_run_id
        )

    def _format_results(
        self,
        result: Union[dict, ComparisonEvaluationResult],
        source_run_id: uuid.UUID,
    ) -> ComparisonEvaluationResult:
        if isinstance(result, ComparisonEvaluationResult):
            if not result.source_run_id:
                result.source_run_id = source_run_id
            return result
        try:
            return ComparisonEvaluationResult(
                **{"source_run_id": source_run_id, **result}
            )
        except ValidationError as e:
            raise ValueError(
                f"Expected a dictionary with a 'key' and dictionary of scores mapping"
                "run IDs to numeric scores, or ComparisonEvaluationResult object,"
                f" got {result}"
            ) from e


def comparison_evaluator(
    func: Callable[
        [Sequence[Run], Optional[Example]],
        Union[_COMPARISON_OUTPUT, Awaitable[_COMPARISON_OUTPUT]],
    ],
) -> DynamicComparisonRunEvaluator:
    """Create a comaprison evaluator from a function."""
    return DynamicComparisonRunEvaluator(func)