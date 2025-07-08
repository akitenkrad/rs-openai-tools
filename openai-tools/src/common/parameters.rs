use serde::{Deserialize, Serialize};
use serde_json::Value;
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

impl From<Value> for ParameterProp {
    fn from(value: Value) -> Self {
        let props = value.as_object().unwrap();
        Self {
            type_name: props.get("type").and_then(Value::as_str).unwrap_or_default().to_string(),
            description: props.get("description").and_then(Value::as_str).map(|s| s.to_string()),
            enum_values: props.get("enum").and_then(Value::as_array).map(|arr| arr.iter().filter_map(Value::as_str).map(|s| s.to_string()).collect()),
        }
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
