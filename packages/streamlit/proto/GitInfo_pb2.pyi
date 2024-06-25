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
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class GitInfo(google.protobuf.message.Message):
    """Message used to update page metadata."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _GitStates:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _GitStatesEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[GitInfo._GitStates.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        DEFAULT: GitInfo._GitStates.ValueType  # 0
        HEAD_DETACHED: GitInfo._GitStates.ValueType  # 1
        AHEAD_OF_REMOTE: GitInfo._GitStates.ValueType  # 2

    class GitStates(_GitStates, metaclass=_GitStatesEnumTypeWrapper): ...
    DEFAULT: GitInfo.GitStates.ValueType  # 0
    HEAD_DETACHED: GitInfo.GitStates.ValueType  # 1
    AHEAD_OF_REMOTE: GitInfo.GitStates.ValueType  # 2

    REPOSITORY_FIELD_NUMBER: builtins.int
    BRANCH_FIELD_NUMBER: builtins.int
    MODULE_FIELD_NUMBER: builtins.int
    UNTRACKED_FILES_FIELD_NUMBER: builtins.int
    UNCOMMITTED_FILES_FIELD_NUMBER: builtins.int
    STATE_FIELD_NUMBER: builtins.int
    repository: builtins.str
    branch: builtins.str
    module: builtins.str
    state: global___GitInfo.GitStates.ValueType
    @property
    def untracked_files(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    @property
    def uncommitted_files(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    def __init__(
        self,
        *,
        repository: builtins.str = ...,
        branch: builtins.str = ...,
        module: builtins.str = ...,
        untracked_files: collections.abc.Iterable[builtins.str] | None = ...,
        uncommitted_files: collections.abc.Iterable[builtins.str] | None = ...,
        state: global___GitInfo.GitStates.ValueType = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["branch", b"branch", "module", b"module", "repository", b"repository", "state", b"state", "uncommitted_files", b"uncommitted_files", "untracked_files", b"untracked_files"]) -> None: ...

global___GitInfo = GitInfo
