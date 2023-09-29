## Value To Value 1.5
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from typing import Literal, Optional

import numpy as np

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)


@invocation_output("str_flt_int_output")
class StrFltIntOutput(BaseInvocationOutput):
    """Output a string, float, and integer"""

    string_output: Optional[str] = OutputField(default="", description="The output string", title="String")
    float_output: Optional[float] = OutputField(default=0, description="The output float", title="Float")
    integer_output: Optional[int] = OutputField(default=0, description="The output integer", title="Integer")


@invocation(
    "value_to_value",
    title="Value To Value",
    tags=["math", "round", "integer", "float", "string", "convert", "v2v"],
    category="math",
    version="1.0.0",
)
class ValueToValueInvocation(BaseInvocation):
    """Outputs a given value as a string, float, or integer. Rounds in the case of float to integer."""

    value: str = InputField(description="The value to pass forward")

    round_multiple: int = InputField(
        default=1, ge=1, title="Multiple of", description="The multiple to round to for float to integer"
    )
    round_method: Literal["Nearest", "Floor", "Ceiling", "Truncate"] = InputField(
        default="Nearest", description="The method to use for rounding"
    )

    def invoke(self, context: InvocationContext) -> StrFltIntOutput:
        try:
            if float(self.value):
                if self.round_method == "Nearest":
                    int_out = round(float(self.value) / self.round_multiple) * self.round_multiple
                elif self.round_method == "Floor":
                    int_out = np.floor(float(self.value) / self.round_multiple) * self.round_multiple
                elif self.round_method == "Ceiling":
                    int_out = np.ceil(float(self.value) / self.round_multiple) * self.round_multiple
                else:  # self.method == "Truncate"
                    int_out = int(float(self.value) / self.round_multiple) * self.round_multiple
                string_out = str(self.value)
                float_out = float(self.value)
                return StrFltIntOutput(
                    string_output=string_out,
                    float_output=float_out,
                    integer_output=int_out,
                )
            else:
                raise ValueError
        except ValueError:
            string_out = str(self.value)
            return StrFltIntOutput(
                string_output=string_out,
            )
