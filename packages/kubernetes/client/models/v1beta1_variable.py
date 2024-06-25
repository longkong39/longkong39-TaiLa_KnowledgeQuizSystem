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


class V1beta1Variable(object):
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
        'expression': 'str',
        'name': 'str'
    }

    attribute_map = {
        'expression': 'expression',
        'name': 'name'
    }

    def __init__(self, expression=None, name=None, local_vars_configuration=None):  # noqa: E501
        """V1beta1Variable - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._expression = None
        self._name = None
        self.discriminator = None

        self.expression = expression
        self.name = name

    @property
    def expression(self):
        """Gets the expression of this V1beta1Variable.  # noqa: E501

        Expression is the expression that will be evaluated as the value of the variable. The CEL expression has access to the same identifiers as the CEL expressions in Validation.  # noqa: E501

        :return: The expression of this V1beta1Variable.  # noqa: E501
        :rtype: str
        """
        return self._expression

    @expression.setter
    def expression(self, expression):
        """Sets the expression of this V1beta1Variable.

        Expression is the expression that will be evaluated as the value of the variable. The CEL expression has access to the same identifiers as the CEL expressions in Validation.  # noqa: E501

        :param expression: The expression of this V1beta1Variable.  # noqa: E501
        :type: str
        """
        if self.local_vars_configuration.client_side_validation and expression is None:  # noqa: E501
            raise ValueError("Invalid value for `expression`, must not be `None`")  # noqa: E501

        self._expression = expression

    @property
    def name(self):
        """Gets the name of this V1beta1Variable.  # noqa: E501

        Name is the name of the variable. The name must be a valid CEL identifier and unique among all variables. The variable can be accessed in other expressions through `variables` For example, if name is \"foo\", the variable will be available as `variables.foo`  # noqa: E501

        :return: The name of this V1beta1Variable.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this V1beta1Variable.

        Name is the name of the variable. The name must be a valid CEL identifier and unique among all variables. The variable can be accessed in other expressions through `variables` For example, if name is \"foo\", the variable will be available as `variables.foo`  # noqa: E501

        :param name: The name of this V1beta1Variable.  # noqa: E501
        :type: str
        """
        if self.local_vars_configuration.client_side_validation and name is None:  # noqa: E501
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

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
        if not isinstance(other, V1beta1Variable):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, V1beta1Variable):
            return True

        return self.to_dict() != other.to_dict()
