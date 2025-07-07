use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Name = String;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ParameterProp {
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

impl ParameterProp {
    pub fn string<U: AsRef<str>>(description: U) -> Self {
        Self { type_name: "string".to_string(), description: Some(description.as_ref().to_string()), ..Default::default() }
    }
    pub fn number<U: AsRef<str>>(description: U) -> Self {
        Self { type_name: "number".to_string(), description: Some(description.as_ref().to_string()), ..Default::default() }
    }
    pub fn boolean<U: AsRef<str>>(description: U) -> Self {
        Self { type_name: "boolean".to_string(), description: Some(description.as_ref().to_string()), ..Default::default() }
    }
    pub fn integer<U: AsRef<str>>(description: U) -> Self {
        Self { type_name: "integer".to_string(), description: Some(description.as_ref().to_string()), ..Default::default() }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Parameters {
    #[serde(rename = "type")]
    pub type_name: String,
    pub properties: HashMap<Name, ParameterProp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<Name>>,
    #[serde(rename = "additionalProperties", skip_serializing_if = "Option::is_none")]
    pub additional_properties: Option<bool>,
}

impl Parameters {
    pub fn new<T: AsRef<str>>(properties: Vec<(T, ParameterProp)>, additional_properties: Option<bool>) -> Self {
        let props = properties.iter().map(|(k, v)| (k.as_ref().to_string(), v.clone())).collect::<HashMap<String, ParameterProp>>();
        let required = properties.iter().map(|(k, _)| k.as_ref().to_string()).collect::<Vec<_>>();
        Self { type_name: "object".into(), properties: props, required: Some(required), additional_properties }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: Parameters,
    pub strict: bool,
}

impl Function {
    pub fn new<T: AsRef<str>, U: AsRef<str>>(name: T, description: U, parameters: Parameters, strict: bool) -> Self {
        Self { name: name.as_ref().to_string(), description: description.as_ref().to_string(), parameters, strict }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<Function>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Parameters>,
}

impl Tool {
    pub fn mcp(
        server_label: String,
        server_url: String,
        require_approval: String,
        allowed_tools: Vec<String>,
        parameters: Vec<(Name, ParameterProp)>,
    ) -> Self {
        Self {
            type_name: "mcp".into(),
            name: Some("".into()),
            server_label: Some(server_label),
            server_url: Some(server_url),
            require_approval: Some(require_approval),
            allowed_tools: Some(allowed_tools),
            parameters: Some(Parameters::new(parameters, None)),
            ..Default::default()
        }
    }

    pub fn function<T: AsRef<str>, U: AsRef<str>, V: AsRef<str>>(name: T, description: U, parameters: Vec<(V, ParameterProp)>, strict: bool) -> Self {
        Self {
            type_name: "function".into(),
            name: Some(name.as_ref().to_string()),
            function: Some(Function::new(name, description, Parameters::new(parameters, None), strict)),
            ..Default::default()
        }
    }
}
