"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
*!
Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2024)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import builtins
import google.protobuf.descriptor
import google.protobuf.message
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class PageNotFound(google.protobuf.message.Message):
    """Message used to tell the client that the requested page does not exist."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PAGE_NAME_FIELD_NUMBER: builtins.int
    page_name: builtins.str
    def __init__(
        self,
        *,
        page_name: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["page_name", b"page_name"]) -> None: ...

global___PageNotFound = PageNotFound