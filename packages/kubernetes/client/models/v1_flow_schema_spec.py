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


class V1FlowSchemaSpec(object):
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
        'distinguisher_method': 'V1FlowDistinguisherMethod',
        'matching_precedence': 'int',
        'priority_level_configuration': 'V1PriorityLevelConfigurationReference',
        'rules': 'list[V1PolicyRulesWithSubjects]'
    }

    attribute_map = {
        'distinguisher_method': 'distinguisherMethod',
        'matching_precedence': 'matchingPrecedence',
        'priority_level_configuration': 'priorityLevelConfiguration',
        'rules': 'rules'
    }

    def __init__(self, distinguisher_method=None, matching_precedence=None, priority_level_configuration=None, rules=None, local_vars_configuration=None):  # noqa: E501
        """V1FlowSchemaSpec - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._distinguisher_method = None
        self._matching_precedence = None
        self._priority_level_configuration = None
        self._rules = None
        self.discriminator = None

        if distinguisher_method is not None:
            self.distinguisher_method = distinguisher_method
        if matching_precedence is not None:
            self.matching_precedence = matching_precedence
        self.priority_level_configuration = priority_level_configuration
        if rules is not None:
            self.rules = rules

    @property
    def distinguisher_method(self):
        """Gets the distinguisher_method of this V1FlowSchemaSpec.  # noqa: E501


        :return: The distinguisher_method of this V1FlowSchemaSpec.  # noqa: E501
        :rtype: V1FlowDistinguisherMethod
        """
        return self._distinguisher_method

    @distinguisher_method.setter
    def distinguisher_method(self, distinguisher_method):
        """Sets the distinguisher_method of this V1FlowSchemaSpec.


        :param distinguisher_method: The distinguisher_method of this V1FlowSchemaSpec.  # noqa: E501
        :type: V1FlowDistinguisherMethod
        """

        self._distinguisher_method = distinguisher_method

    @property
    def matching_precedence(self):
        """Gets the matching_precedence of this V1FlowSchemaSpec.  # noqa: E501

        `matchingPrecedence` is used to choose among the FlowSchemas that match a given request. The chosen FlowSchema is among those with the numerically lowest (which we take to be logically highest) MatchingPrecedence.  Each MatchingPrecedence value must be ranged in [1,10000]. Note that if the precedence is not specified, it will be set to 1000 as default.  # noqa: E501

        :return: The matching_precedence of this V1FlowSchemaSpec.  # noqa: E501
        :rtype: int
        """
        return self._matching_precedence

    @matching_precedence.setter
    def matching_precedence(self, matching_precedence):
        """Sets the matching_precedence of this V1FlowSchemaSpec.

        `matchingPrecedence` is used to choose among the FlowSchemas that match a given request. The chosen FlowSchema is among those with the numerically lowest (which we take to be logically highest) MatchingPrecedence.  Each MatchingPrecedence value must be ranged in [1,10000]. Note that if the precedence is not specified, it will be set to 1000 as default.  # noqa: E501

        :param matching_precedence: The matching_precedence of this V1FlowSchemaSpec.  # noqa: E501
        :type: int
        """

        self._matching_precedence = matching_precedence

    @property
    def priority_level_configuration(self):
        """Gets the priority_level_configuration of this V1FlowSchemaSpec.  # noqa: E501


        :return: The priority_level_configuration of this V1FlowSchemaSpec.  # noqa: E501
        :rtype: V1PriorityLevelConfigurationReference
        """
        return self._priority_level_configuration

    @priority_level_configuration.setter
    def priority_level_configuration(self, priority_level_configuration):
        """Sets the priority_level_configuration of this V1FlowSchemaSpec.


        :param priority_level_configuration: The priority_level_configuration of this V1FlowSchemaSpec.  # noqa: E501
        :type: V1PriorityLevelConfigurationReference
        """
        if self.local_vars_configuration.client_side_validation and priority_level_configuration is None:  # noqa: E501
            raise ValueError("Invalid value for `priority_level_configuration`, must not be `None`")  # noqa: E501

        self._priority_level_configuration = priority_level_configuration

    @property
    def rules(self):
        """Gets the rules of this V1FlowSchemaSpec.  # noqa: E501

        `rules` describes which requests will match this flow schema. This FlowSchema matches a request if and only if at least one member of rules matches the request. if it is an empty slice, there will be no requests matching the FlowSchema.  # noqa: E501

        :return: The rules of this V1FlowSchemaSpec.  # noqa: E501
        :rtype: list[V1PolicyRulesWithSubjects]
        """
        return self._rules

    @rules.setter
    def rules(self, rules):
        """Sets the rules of this V1FlowSchemaSpec.

        `rules` describes which requests will match this flow schema. This FlowSchema matches a request if and only if at least one member of rules matches the request. if it is an empty slice, there will be no requests matching the FlowSchema.  # noqa: E501

        :param rules: The rules of this V1FlowSchemaSpec.  # noqa: E501
        :type: list[V1PolicyRulesWithSubjects]
        """

        self._rules = rules

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
        if not isinstance(other, V1FlowSchemaSpec):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, V1FlowSchemaSpec):
            return True

        return self.to_dict() != other.to_dict()
