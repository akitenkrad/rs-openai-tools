use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub type Name = String;

#[derive(Debug, Clone, Default)]
pub struct ParameterProperty {
    type_names: Vec<String>,
    pub description: Option<String>,
    enum_values: Vec<String>,
}

impl From<Value> for ParameterProperty {
    fn from(value: Value) -> Self {
        let props = value.as_object().unwrap();
        let type_names = {
            if let Some(type_name) = props.get("type") {
                if let Some(type_array) = type_name.as_array() {
                    type_array.iter().filter_map(Value::as_str).map(|s| s.to_string()).collect()
                } else if let Some(type_str) = type_name.as_str() {
                    vec![type_str.to_string()]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        };
        let description = props.get("description").and_then(Value::as_str).map(|s| s.to_string());
        let enum_values = {
            if let Some(enum_value) = props.get("enum") {
                if let Some(enum_array) = enum_value.as_array() {
                    enum_array.iter().filter_map(Value::as_str).map(|s| s.to_string()).collect()
                } else if let Some(enum_str) = enum_value.as_str() {
                    vec![enum_str.to_string()]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        };
        Self { type_names, description, enum_values }
    }
}

impl From<ParameterProperty> for Value {
    fn from(prop: ParameterProperty) -> Self {
        let mut map = serde_json::Map::new();

        // type
        if prop.type_names.len() == 1 {
            map.insert("type".to_string(), Value::String(prop.type_names[0].clone()));
        } else {
            map.insert("type".to_string(), Value::Array(prop.type_names.iter().map(|s| Value::String(s.clone())).collect()));
        }

        // description
        if let Some(desc) = &prop.description {
            map.insert("description".to_string(), Value::String(desc.clone()));
        }

        // enum
        if prop.enum_values.len() == 1 {
            map.insert("enum".to_string(), Value::String(prop.enum_values[0].clone()));
        } else if prop.enum_values.len() > 1 {
            map.insert("enum".to_string(), Value::Array(prop.enum_values.iter().map(|s| Value::String(s.clone())).collect()));
        }
        Value::Object(map)
    }
}

impl Serialize for ParameterProperty {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Assert
        if self.type_names.is_empty() {
            return Err(serde::ser::Error::custom("Type name is required"));
        }
        if self.enum_values.len() > 0 && self.enum_values.len() != self.type_names.len() {
            return Err(serde::ser::Error::custom("Enum values must match type names count"));
        }

        // Serialize
        let value = Value::from(self.clone());
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ParameterProperty {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value: Value = Deserialize::deserialize(deserializer)?;
        Ok(ParameterProperty::from(value))
    }
}

impl ParameterProperty {
    pub fn new<T: AsRef<str>>(description: T) -> Self {
        Self { type_names: vec![], description: Some(description.as_ref().to_string()), enum_values: Vec::new() }
    }

    pub fn add_string(&mut self) -> Self {
        self.type_names.push("string".into());
        self.clone()
    }
    pub fn add_number(&mut self) -> Self {
        self.type_names.push("number".into());
        self.clone()
    }
    pub fn add_boolean(&mut self) -> Self {
        self.type_names.push("boolean".into());
        self.clone()
    }
    pub fn add_integer(&mut self) -> Self {
        self.type_names.push("integer".into());
        self.clone()
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Parameters {
    #[serde(rename = "type")]
    pub type_name: String,
    pub properties: HashMap<Name, ParameterProperty>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<Name>>,
    #[serde(rename = "additionalProperties", skip_serializing_if = "Option::is_none")]
    pub additional_properties: Option<bool>,
}

impl Parameters {
    pub fn new<T: AsRef<str>>(properties: Vec<(T, ParameterProperty)>, additional_properties: Option<bool>) -> Self {
        let props = properties.iter().map(|(k, v)| (k.as_ref().to_string(), v.clone())).collect::<HashMap<String, ParameterProperty>>();
        let required = properties.iter().map(|(k, _)| k.as_ref().to_string()).collect::<Vec<_>>();
        Self { type_name: "object".into(), properties: props, required: Some(required), additional_properties }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_property_serialization_1() {
        let prop = ParameterProperty::new("A string parameter").add_string();
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_names, vec!["string".to_string()]);
        assert_eq!(deserialized.description, Some("A string parameter".to_string()));

        let prop = ParameterProperty::new("An integer parameter").add_integer();
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_names, vec!["integer".to_string()]);
        assert_eq!(deserialized.description, Some("An integer parameter".to_string()));

        let prop = ParameterProperty::new("A boolean parameter").add_boolean();
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_names, vec!["boolean".to_string()]);
        assert_eq!(deserialized.description, Some("A boolean parameter".to_string()));

        let prop = ParameterProperty::new("A number parameter").add_number();
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_names, vec!["number".to_string()]);
        assert_eq!(deserialized.description, Some("A number parameter".to_string()));
    }

    #[test]
    fn test_parameter_property_serialization_2() {
        let prop = ParameterProperty::new("A string or number parameter").add_string().add_number();
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_names, vec!["string".to_string(), "number".to_string()]);
        assert_eq!(deserialized.description, Some("A string or number parameter".to_string()));

        let prop = ParameterProperty::new("An enum parameter").add_string().add_integer().add_boolean().add_number();
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_names, vec!["string".to_string(), "integer".to_string(), "boolean".to_string(), "number".to_string()]);
        assert_eq!(deserialized.description, Some("An enum parameter".to_string()));
    }
}
