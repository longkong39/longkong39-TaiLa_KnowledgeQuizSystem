# coding: utf-8

"""
    Kubernetes

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: release-1.30
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six

from kubernetes.client.configuration import Configuration


class V1EphemeralContainer(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'args': 'list[str]',
        'command': 'list[str]',
        'env': 'list[V1EnvVar]',
        'env_from': 'list[V1EnvFromSource]',
        'image': 'str',
        'image_pull_policy': 'str',
        'lifecycle': 'V1Lifecycle',
        'liveness_probe': 'V1Probe',
        'name': 'str',
        'ports': 'list[V1ContainerPort]',
        'readiness_probe': 'V1Probe',
        'resize_policy': 'list[V1ContainerResizePolicy]',
        'resources': 'V1ResourceRequirements',
        'restart_policy': 'str',
        'security_context': 'V1SecurityContext',
        'startup_probe': 'V1Probe',
        'stdin': 'bool',
        'stdin_once': 'bool',
        'target_container_name': 'str',
        'termination_message_path': 'str',
        'termination_message_policy': 'str',
        'tty': 'bool',
        'volume_devices': 'list[V1VolumeDevice]',
        'volume_mounts': 'list[V1VolumeMount]',
        'working_dir': 'str'
    }

    attribute_map = {
        'args': 'args',
        'command': 'command',
        'env': 'env',
        'env_from': 'envFrom',
        'image': 'image',
        'image_pull_policy': 'imagePullPolicy',
        'lifecycle': 'lifecycle',
        'liveness_probe': 'livenessProbe',
        'name': 'name',
        'ports': 'ports',
        'readiness_probe': 'readinessProbe',
        'resize_policy': 'resizePolicy',
        'resources': 'resources',
        'restart_policy': 'restartPolicy',
        'security_context': 'securityContext',
        'startup_probe': 'startupProbe',
        'stdin': 'stdin',
        'stdin_once': 'stdinOnce',
        'target_container_name': 'targetContainerName',
        'termination_message_path': 'terminationMessagePath',
        'termination_message_policy': 'terminationMessagePolicy',
        'tty': 'tty',
        'volume_devices': 'volumeDevices',
        'volume_mounts': 'volumeMounts',
        'working_dir': 'workingDir'
    }

    def __init__(self, args=None, command=None, env=None, env_from=None, image=None, image_pull_policy=None, lifecycle=None, liveness_probe=None, name=None, ports=None, readiness_probe=None, resize_policy=None, resources=None, restart_policy=None, security_context=None, startup_probe=None, stdin=None, stdin_once=None, target_container_name=None, termination_message_path=None, termination_message_policy=None, tty=None, volume_devices=None, volume_mounts=None, working_dir=None, local_vars_configuration=None):  # noqa: E501
        """V1EphemeralContainer - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._args = None
        self._command = None
        self._env = None
        self._env_from = None
        self._image = None
        self._image_pull_policy = None
        self._lifecycle = None
        self._liveness_probe = None
        self._name = None
        self._ports = None
        self._readiness_probe = None
        self._resize_policy = None
        self._resources = None
        self._restart_policy = None
        self._security_context = None
        self._startup_probe = None
        self._stdin = None
        self._stdin_once = None
        self._target_container_name = None
        self._termination_message_path = None
        self._termination_message_policy = None
        self._tty = None
        self._volume_devices = None
        self._volume_mounts = None
        self._working_dir = None
        self.discriminator = None

        if args is not None:
            self.args = args
        if command is not None:
            self.command = command
        if env is not None:
            self.env = env
        if env_from is not None:
            self.env_from = env_from
        if image is not None:
            self.image = image
        if image_pull_policy is not None:
            self.image_pull_policy = image_pull_policy
        if lifecycle is not None:
            self.lifecycle = lifecycle
        if liveness_probe is not None:
            self.liveness_probe = liveness_probe
        self.name = name
        if ports is not None:
            self.ports = ports
        if readiness_probe is not None:
            self.readiness_probe = readiness_probe
        if resize_policy is not None:
            self.resize_policy = resize_policy
        if resources is not None:
            self.resources = resources
        if restart_policy is not None:
            self.restart_policy = restart_policy
        if security_context is not None:
            self.security_context = security_context
        if startup_probe is not None:
            self.startup_probe = startup_probe
        if stdin is not None:
            self.stdin = stdin
        if stdin_once is not None:
            self.stdin_once = stdin_once
        if target_container_name is not None:
            self.target_container_name = target_container_name
        if termination_message_path is not None:
            self.termination_message_path = termination_message_path
        if termination_message_policy is not None:
            self.termination_message_policy = termination_message_policy
        if tty is not None:
            self.tty = tty
        if volume_devices is not None:
            self.volume_devices = volume_devices
        if volume_mounts is not None:
            self.volume_mounts = volume_mounts
        if working_dir is not None:
            self.working_dir = working_dir

    @property
    def args(self):
        """Gets the args of this V1EphemeralContainer.  # noqa: E501

        Arguments to the entrypoint. The image's CMD is used if this is not provided. Variable references $(VAR_NAME) are expanded using the container's environment. If a variable cannot be resolved, the reference in the input string will be unchanged. Double $$ are reduced to a single $, which allows for escaping the $(VAR_NAME) syntax: i.e. \"$$(VAR_NAME)\" will produce the string literal \"$(VAR_NAME)\". Escaped references will never be expanded, regardless of whether the variable exists or not. Cannot be updated. More info: https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#running-a-command-in-a-shell  # noqa: E501

        :return: The args of this V1EphemeralContainer.  # noqa: E501
        :rtype: list[str]
        """
        return self._args

    @args.setter
    def args(self, args):
        """Sets the args of this V1EphemeralContainer.

        Arguments to the entrypoint. The image's CMD is used if this is not provided. Variable references $(VAR_NAME) are expanded using the container's environment. If a variable cannot be resolved, the reference in the input string will be unchanged. Double $$ are reduced to a single $, which allows for escaping the $(VAR_NAME) syntax: i.e. \"$$(VAR_NAME)\" will produce the string literal \"$(VAR_NAME)\". Escaped references will never be expanded, regardless of whether the variable exists or not. Cannot be updated. More info: https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#running-a-command-in-a-shell  # noqa: E501

        :param args: The args of this V1EphemeralContainer.  # noqa: E501
        :type: list[str]
        """

        self._args = args

    @property
    def command(self):
        """Gets the command of this V1EphemeralContainer.  # noqa: E501

        Entrypoint array. Not executed within a shell. The image's ENTRYPOINT is used if this is not provided. Variable references $(VAR_NAME) are expanded using the container's environment. If a variable cannot be resolved, the reference in the input string will be unchanged. Double $$ are reduced to a single $, which allows for escaping the $(VAR_NAME) syntax: i.e. \"$$(VAR_NAME)\" will produce the string literal \"$(VAR_NAME)\". Escaped references will never be expanded, regardless of whether the variable exists or not. Cannot be updated. More info: https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#running-a-command-in-a-shell  # noqa: E501

        :return: The command of this V1EphemeralContainer.  # noqa: E501
        :rtype: list[str]
        """
        return self._command

    @command.setter
    def command(self, command):
        """Sets the command of this V1EphemeralContainer.

        Entrypoint array. Not executed within a shell. The image's ENTRYPOINT is used if this is not provided. Variable references $(VAR_NAME) are expanded using the container's environment. If a variable cannot be resolved, the reference in the input string will be unchanged. Double $$ are reduced to a single $, which allows for escaping the $(VAR_NAME) syntax: i.e. \"$$(VAR_NAME)\" will produce the string literal \"$(VAR_NAME)\". Escaped references will never be expanded, regardless of whether the variable exists or not. Cannot be updated. More info: https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#running-a-command-in-a-shell  # noqa: E501

        :param command: The command of this V1EphemeralContainer.  # noqa: E501
        :type: list[str]
        """

        self._command = command

    @property
    def env(self):
        """Gets the env of this V1EphemeralContainer.  # noqa: E501

        List of environment variables to set in the container. Cannot be updated.  # noqa: E501

        :return: The env of this V1EphemeralContainer.  # noqa: E501
        :rtype: list[V1EnvVar]
        """
        return self._env

    @env.setter
    def env(self, env):
        """Sets the env of this V1EphemeralContainer.

        List of environment variables to set in the container. Cannot be updated.  # noqa: E501

        :param env: The env of this V1EphemeralContainer.  # noqa: E501
        :type: list[V1EnvVar]
        """

        self._env = env

    @property
    def env_from(self):
        """Gets the env_from of this V1EphemeralContainer.  # noqa: E501

        List of sources to populate environment variables in the container. The keys defined within a source must be a C_IDENTIFIER. All invalid keys will be reported as an event when the container is starting. When a key exists in multiple sources, the value associated with the last source will take precedence. Values defined by an Env with a duplicate key will take precedence. Cannot be updated.  # noqa: E501

        :return: The env_from of this V1EphemeralContainer.  # noqa: E501
        :rtype: list[V1EnvFromSource]
        """
        return self._env_from

    @env_from.setter
    def env_from(self, env_from):
        """Sets the env_from of this V1EphemeralContainer.

        List of sources to populate environment variables in the container. The keys defined within a source must be a C_IDENTIFIER. All invalid keys will be reported as an event when the container is starting. When a key exists in multiple sources, the value associated with the last source will take precedence. Values defined by an Env with a duplicate key will take precedence. Cannot be updated.  # noqa: E501

        :param env_from: The env_from of this V1EphemeralContainer.  # noqa: E501
        :type: list[V1EnvFromSource]
        """

        self._env_from = env_from

    @property
    def image(self):
        """Gets the image of this V1EphemeralContainer.  # noqa: E501

        Container image name. More info: https://kubernetes.io/docs/concepts/containers/images  # noqa: E501

        :return: The image of this V1EphemeralContainer.  # noqa: E501
        :rtype: str
        """
        return self._image

    @image.setter
    def image(self, image):
        """Sets the image of this V1EphemeralContainer.

        Container image name. More info: https://kubernetes.io/docs/concepts/containers/images  # noqa: E501

        :param image: The image of this V1EphemeralContainer.  # noqa: E501
        :type: str
        """

        self._image = image

    @property
    def image_pull_policy(self):
        """Gets the image_pull_policy of this V1EphemeralContainer.  # noqa: E501

        Image pull policy. One of Always, Never, IfNotPresent. Defaults to Always if :latest tag is specified, or IfNotPresent otherwise. Cannot be updated. More info: https://kubernetes.io/docs/concepts/containers/images#updating-images  # noqa: E501

        :return: The image_pull_policy of this V1EphemeralContainer.  # noqa: E501
        :rtype: str
        """
        return self._image_pull_policy

    @image_pull_policy.setter
    def image_pull_policy(self, image_pull_policy):
        """Sets the image_pull_policy of this V1EphemeralContainer.

        Image pull policy. One of Always, Never, IfNotPresent. Defaults to Always if :latest tag is specified, or IfNotPresent otherwise. Cannot be updated. More info: https://kubernetes.io/docs/concepts/containers/images#updating-images  # noqa: E501

        :param image_pull_policy: The image_pull_policy of this V1EphemeralContainer.  # noqa: E501
        :type: str
        """

        self._image_pull_policy = image_pull_policy

    @property
    def lifecycle(self):
        """Gets the lifecycle of this V1EphemeralContainer.  # noqa: E501


        :return: The lifecycle of this V1EphemeralContainer.  # noqa: E501
        :rtype: V1Lifecycle
        """
        return self._lifecycle

    @lifecycle.setter
    def lifecycle(self, lifecycle):
        """Sets the lifecycle of this V1EphemeralContainer.


        :param lifecycle: The lifecycle of this V1EphemeralContainer.  # noqa: E501
        :type: V1Lifecycle
        """

        self._lifecycle = lifecycle

    @property
    def liveness_probe(self):
        """Gets the liveness_probe of this V1EphemeralContainer.  # noqa: E501


        :return: The liveness_probe of this V1EphemeralContainer.  # noqa: E501
        :rtype: V1Probe
        """
        return self._liveness_probe

    @liveness_probe.setter
    def liveness_probe(self, liveness_probe):
        """Sets the liveness_probe of this V1EphemeralContainer.


        :param liveness_probe: The liveness_probe of this V1EphemeralContainer.  # noqa: E501
        :type: V1Probe
        """

        self._liveness_probe = liveness_probe

    @property
    def name(self):
        """Gets the name of this V1EphemeralContainer.  # noqa: E501

        Name of the ephemeral container specified as a DNS_LABEL. This name must be unique among all containers, init containers and ephemeral containers.  # noqa: E501

        :return: The name of this V1EphemeralContainer.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this V1EphemeralContainer.

        Name of the ephemeral container specified as a DNS_LABEL. This name must be unique among all containers, init containers and ephemeral containers.  # noqa: E501

        :param name: The name of this V1EphemeralContainer.  # noqa: E501
        :type: str
        """
        if self.local_vars_configuration.client_side_validation and name is None:  # noqa: E501
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def ports(self):
        """Gets the ports of this V1EphemeralContainer.  # noqa: E501

        Ports are not allowed for ephemeral containers.  # noqa: E501

        :return: The ports of this V1EphemeralContainer.  # noqa: E501
        :rtype: list[V1ContainerPort]
        """
        return self._ports

    @ports.setter
    def ports(self, ports):
        """Sets the ports of this V1EphemeralContainer.

        Ports are not allowed for ephemeral containers.  # noqa: E501

        :param ports: The ports of this V1EphemeralContainer.  # noqa: E501
        :type: list[V1ContainerPort]
        """

        self._ports = ports

    @property
    def readiness_probe(self):
        """Gets the readiness_probe of this V1EphemeralContainer.  # noqa: E501


        :return: The readiness_probe of this V1EphemeralContainer.  # noqa: E501
        :rtype: V1Probe
        """
        return self._readiness_probe

    @readiness_probe.setter
    def readiness_probe(self, readiness_probe):
        """Sets the readiness_probe of this V1EphemeralContainer.


        :param readiness_probe: The readiness_probe of this V1EphemeralContainer.  # noqa: E501
        :type: V1Probe
        """

        self._readiness_probe = readiness_probe

    @property
    def resize_policy(self):
        """Gets the resize_policy of this V1EphemeralContainer.  # noqa: E501

        Resources resize policy for the container.  # noqa: E501

        :return: The resize_policy of this V1EphemeralContainer.  # noqa: E501
        :rtype: list[V1ContainerResizePolicy]
        """
        return self._resize_policy

    @resize_policy.setter
    def resize_policy(self, resize_policy):
        """Sets the resize_policy of this V1EphemeralContainer.

        Resources resize policy for the container.  # noqa: E501

        :param resize_policy: The resize_policy of this V1EphemeralContainer.  # noqa: E501
        :type: list[V1ContainerResizePolicy]
        """

        self._resize_policy = resize_policy

    @property
    def resources(self):
        """Gets the resources of this V1EphemeralContainer.  # noqa: E501


        :return: The resources of this V1EphemeralContainer.  # noqa: E501
        :rtype: V1ResourceRequirements
        """
        return self._resources

    @resources.setter
    def resources(self, resources):
        """Sets the resources of this V1EphemeralContainer.


        :param resources: The resources of this V1EphemeralContainer.  # noqa: E501
        :type: V1ResourceRequirements
        """

        self._resources = resources

    @property
    def restart_policy(self):
        """Gets the restart_policy of this V1EphemeralContainer.  # noqa: E501

        Restart policy for the container to manage the restart behavior of each container within a pod. This may only be set for init containers. You cannot set this field on ephemeral containers.  # noqa: E501

        :return: The restart_policy of this V1EphemeralContainer.  # noqa: E501
        :rtype: str
        """
        return self._restart_policy

    @restart_policy.setter
    def restart_policy(self, restart_policy):
        """Sets the restart_policy of this V1EphemeralContainer.

        Restart policy for the container to manage the restart behavior of each container within a pod. This may only be set for init containers. You cannot set this field on ephemeral containers.  # noqa: E501

        :param restart_policy: The restart_policy of this V1EphemeralContainer.  # noqa: E501
        :type: str
        """

        self._restart_policy = restart_policy

    @property
    def security_context(self):
        """Gets the security_context of this V1EphemeralContainer.  # noqa: E501


        :return: The security_context of this V1EphemeralContainer.  # noqa: E501
        :rtype: V1SecurityContext
        """
        return self._security_context

    @security_context.setter
    def security_context(self, security_context):
        """Sets the security_context of this V1EphemeralContainer.


        :param security_context: The security_context of this V1EphemeralContainer.  # noqa: E501
        :type: V1SecurityContext
        """

        self._security_context = security_context

    @property
    def startup_probe(self):
        """Gets the startup_probe of this V1EphemeralContainer.  # noqa: E501


        :return: The startup_probe of this V1EphemeralContainer.  # noqa: E501
        :rtype: V1Probe
        """
        return self._startup_probe

    @startup_probe.setter
    def startup_probe(self, startup_probe):
        """Sets the startup_probe of this V1EphemeralContainer.


        :param startup_probe: The startup_probe of this V1EphemeralContainer.  # noqa: E501
        :type: V1Probe
        """

        self._startup_probe = startup_probe

    @property
    def stdin(self):
        """Gets the stdin of this V1EphemeralContainer.  # noqa: E501

        Whether this container should allocate a buffer for stdin in the container runtime. If this is not set, reads from stdin in the container will always result in EOF. Default is false.  # noqa: E501

        :return: The stdin of this V1EphemeralContainer.  # noqa: E501
        :rtype: bool
        """
        return self._stdin

    @stdin.setter
    def stdin(self, stdin):
        """Sets the stdin of this V1EphemeralContainer.

        Whether this container should allocate a buffer for stdin in the container runtime. If this is not set, reads from stdin in the container will always result in EOF. Default is false.  # noqa: E501

        :param stdin: The stdin of this V1EphemeralContainer.  # noqa: E501
        :type: bool
        """

        self._stdin = stdin

    @property
    def stdin_once(self):
        """Gets the stdin_once of this V1EphemeralContainer.  # noqa: E501

        Whether the container runtime should close the stdin channel after it has been opened by a single attach. When stdin is true the stdin stream will remain open across multiple attach sessions. If stdinOnce is set to true, stdin is opened on container start, is empty until the first client attaches to stdin, and then remains open and accepts data until the client disconnects, at which time stdin is closed and remains closed until the container is restarted. If this flag is false, a container processes that reads from stdin will never receive an EOF. Default is false  # noqa: E501

        :return: The stdin_once of this V1EphemeralContainer.  # noqa: E501
        :rtype: bool
        """
        return self._stdin_once

    @stdin_once.setter
    def stdin_once(self, stdin_once):
        """Sets the stdin_once of this V1EphemeralContainer.

        Whether the container runtime should close the stdin channel after it has been opened by a single attach. When stdin is true the stdin stream will remain open across multiple attach sessions. If stdinOnce is set to true, stdin is opened on container start, is empty until the first client attaches to stdin, and then remains open and accepts data until the client disconnects, at which time stdin is closed and remains closed until the container is restarted. If this flag is false, a container processes that reads from stdin will never receive an EOF. Default is false  # noqa: E501

        :param stdin_once: The stdin_once of this V1EphemeralContainer.  # noqa: E501
        :type: bool
        """

        self._stdin_once = stdin_once

    @property
    def target_container_name(self):
        """Gets the target_container_name of this V1EphemeralContainer.  # noqa: E501

        If set, the name of the container from PodSpec that this ephemeral container targets. The ephemeral container will be run in the namespaces (IPC, PID, etc) of this container. If not set then the ephemeral container uses the namespaces configured in the Pod spec.  The container runtime must implement support for this feature. If the runtime does not support namespace targeting then the result of setting this field is undefined.  # noqa: E501

        :return: The target_container_name of this V1EphemeralContainer.  # noqa: E501
        :rtype: str
        """
        return self._target_container_name

    @target_container_name.setter
    def target_container_name(self, target_container_name):
        """Sets the target_container_name of this V1EphemeralContainer.

        If set, the name of the container from PodSpec that this ephemeral container targets. The ephemeral container will be run in the namespaces (IPC, PID, etc) of this container. If not set then the ephemeral container uses the namespaces configured in the Pod spec.  The container runtime must implement support for this feature. If the runtime does not support namespace targeting then the result of setting this field is undefined.  # noqa: E501

        :param target_container_name: The target_container_name of this V1EphemeralContainer.  # noqa: E501
        :type: str
        """

        self._target_container_name = target_container_name

    @property
    def termination_message_path(self):
        """Gets the termination_message_path of this V1EphemeralContainer.  # noqa: E501

        Optional: Path at which the file to which the container's termination message will be written is mounted into the container's filesystem. Message written is intended to be brief final status, such as an assertion failure message. Will be truncated by the node if greater than 4096 bytes. The total message length across all containers will be limited to 12kb. Defaults to /dev/termination-log. Cannot be updated.  # noqa: E501

        :return: The termination_message_path of this V1EphemeralContainer.  # noqa: E501
        :rtype: str
        """
        return self._termination_message_path

    @termination_message_path.setter
    def termination_message_path(self, termination_message_path):
        """Sets the termination_message_path of this V1EphemeralContainer.

        Optional: Path at which the file to which the container's termination message will be written is mounted into the container's filesystem. Message written is intended to be brief final status, such as an assertion failure message. Will be truncated by the node if greater than 4096 bytes. The total message length across all containers will be limited to 12kb. Defaults to /dev/termination-log. Cannot be updated.  # noqa: E501

        :param termination_message_path: The termination_message_path of this V1EphemeralContainer.  # noqa: E501
        :type: str
        """

        self._termination_message_path = termination_message_path

    @property
    def termination_message_policy(self):
        """Gets the termination_message_policy of this V1EphemeralContainer.  # noqa: E501

        Indicate how the termination message should be populated. File will use the contents of terminationMessagePath to populate the container status message on both success and failure. FallbackToLogsOnError will use the last chunk of container log output if the termination message file is empty and the container exited with an error. The log output is limited to 2048 bytes or 80 lines, whichever is smaller. Defaults to File. Cannot be updated.  # noqa: E501

        :return: The termination_message_policy of this V1EphemeralContainer.  # noqa: E501
        :rtype: str
        """
        return self._termination_message_policy

    @termination_message_policy.setter
    def termination_message_policy(self, termination_message_policy):
        """Sets the termination_message_policy of this V1EphemeralContainer.

        Indicate how the termination message should be populated. File will use the contents of terminationMessagePath to populate the container status message on both success and failure. FallbackToLogsOnError will use the last chunk of container log output if the termination message file is empty and the container exited with an error. The log output is limited to 2048 bytes or 80 lines, whichever is smaller. Defaults to File. Cannot be updated.  # noqa: E501

        :param termination_message_policy: The termination_message_policy of this V1EphemeralContainer.  # noqa: E501
        :type: str
        """

        self._termination_message_policy = termination_message_policy

    @property
    def tty(self):
        """Gets the tty of this V1EphemeralContainer.  # noqa: E501

        Whether this container should allocate a TTY for itself, also requires 'stdin' to be true. Default is false.  # noqa: E501

        :return: The tty of this V1EphemeralContainer.  # noqa: E501
        :rtype: bool
        """
        return self._tty

    @tty.setter
    def tty(self, tty):
        """Sets the tty of this V1EphemeralContainer.

        Whether this container should allocate a TTY for itself, also requires 'stdin' to be true. Default is false.  # noqa: E501

        :param tty: The tty of this V1EphemeralContainer.  # noqa: E501
        :type: bool
        """

        self._tty = tty

    @property
    def volume_devices(self):
        """Gets the volume_devices of this V1EphemeralContainer.  # noqa: E501

        volumeDevices is the list of block devices to be used by the container.  # noqa: E501

        :return: The volume_devices of this V1EphemeralContainer.  # noqa: E501
        :rtype: list[V1VolumeDevice]
        """
        return self._volume_devices

    @volume_devices.setter
    def volume_devices(self, volume_devices):
        """Sets the volume_devices of this V1EphemeralContainer.

        volumeDevices is the list of block devices to be used by the container.  # noqa: E501

        :param volume_devices: The volume_devices of this V1EphemeralContainer.  # noqa: E501
        :type: list[V1VolumeDevice]
        """

        self._volume_devices = volume_devices

    @property
    def volume_mounts(self):
        """Gets the volume_mounts of this V1EphemeralContainer.  # noqa: E501

        Pod volumes to mount into the container's filesystem. Subpath mounts are not allowed for ephemeral containers. Cannot be updated.  # noqa: E501

        :return: The volume_mounts of this V1EphemeralContainer.  # noqa: E501
        :rtype: list[V1VolumeMount]
        """
        return self._volume_mounts

    @volume_mounts.setter
    def volume_mounts(self, volume_mounts):
        """Sets the volume_mounts of this V1EphemeralContainer.

        Pod volumes to mount into the container's filesystem. Subpath mounts are not allowed for ephemeral containers. Cannot be updated.  # noqa: E501

        :param volume_mounts: The volume_mounts of this V1EphemeralContainer.  # noqa: E501
        :type: list[V1VolumeMount]
        """

        self._volume_mounts = volume_mounts

    @property
    def working_dir(self):
        """Gets the working_dir of this V1EphemeralContainer.  # noqa: E501

        Container's working directory. If not specified, the container runtime's default will be used, which might be configured in the container image. Cannot be updated.  # noqa: E501

        :return: The working_dir of this V1EphemeralContainer.  # noqa: E501
        :rtype: str
        """
        return self._working_dir

    @working_dir.setter
    def working_dir(self, working_dir):
        """Sets the working_dir of this V1EphemeralContainer.

        Container's working directory. If not specified, the container runtime's default will be used, which might be configured in the container image. Cannot be updated.  # noqa: E501

        :param working_dir: The working_dir of this V1EphemeralContainer.  # noqa: E501
        :type: str
        """

        self._working_dir = working_dir

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V1EphemeralContainer):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, V1EphemeralContainer):
            return True

        return self.to_dict() != other.to_dict()