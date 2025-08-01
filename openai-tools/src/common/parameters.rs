use serde::{de::Error, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub type Name = String;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ParameterProperty {
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

impl TryFrom<Value> for ParameterProperty {
    type Error = serde_json::Error;
    fn try_from(value: Value) -> Result<Self, serde_json::Error> {
        let props = value.as_object().ok_or_else(|| serde_json::Error::custom("Expected an object"))?;
        let type_name = {
            if let Some(type_name) = props.get("type") {
                if let Some(type_name) = type_name.as_str() {
                    type_name.to_string()
                } else {
                    return Err(serde_json::Error::custom("Expected 'type' to be a string"));
                }
            } else {
                return Err(serde_json::Error::custom("Missing 'type' field"));
            }
        };
        let description = props.get("description").and_then(Value::as_str).map(|s| s.to_string());
        let enum_values = {
            if let Some(enum_value) = props.get("enum") {
                if let Some(enum_array) = enum_value.as_array() {
                    Some(enum_array.iter().filter_map(Value::as_str).map(|s| s.to_string()).collect::<Vec<String>>())
                } else {
                    return Err(serde_json::Error::custom("Expected 'enum' to be an array"));
                }
            } else {
                None
            }
        };
        Ok(Self { type_name, description, enum_values })
    }
}

impl From<ParameterProperty> for Value {
    fn from(prop: ParameterProperty) -> Self {
        let mut map = serde_json::Map::new();

        // type
        map.insert("type".to_string(), Value::String(prop.type_name.clone()));

        // description
        if let Some(desc) = &prop.description {
            map.insert("description".to_string(), Value::String(desc.clone()));
        }

        // enum
        if let Some(enum_values) = prop.enum_values {
            map.insert("enum".to_string(), Value::Array(enum_values.iter().map(|s| Value::String(s.clone())).collect()));
        }
        Value::Object(map)
    }
}

impl ParameterProperty {
    pub fn from_string<T: AsRef<str>>(description: T) -> Self {
        Self { type_name: "string".into(), description: Some(description.as_ref().to_string()), enum_values: None }
    }
    pub fn from_number<T: AsRef<str>>(description: T) -> Self {
        Self { type_name: "number".into(), description: Some(description.as_ref().to_string()), enum_values: None }
    }
    pub fn from_boolean<T: AsRef<str>>(description: T) -> Self {
        Self { type_name: "boolean".into(), description: Some(description.as_ref().to_string()), enum_values: None }
    }
    pub fn from_integer<T: AsRef<str>>(description: T) -> Self {
        Self { type_name: "integer".into(), description: Some(description.as_ref().to_string()), enum_values: None }
    }
    pub fn add_enum_values<T: AsRef<str>>(&mut self, values: Vec<T>) -> Self {
        self.enum_values = Some(values.into_iter().map(|v| v.as_ref().to_string()).collect());
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
    fn test_parameter_property_serialization() {
        let prop = ParameterProperty::from_string("A string parameter");
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_name, "string");
        assert_eq!(deserialized.description, Some("A string parameter".to_string()));

        let prop = ParameterProperty::from_integer("An integer parameter");
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_name, "integer");
        assert_eq!(deserialized.description, Some("An integer parameter".to_string()));

        let prop = ParameterProperty::from_boolean("A boolean parameter");
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_name, "boolean");
        assert_eq!(deserialized.description, Some("A boolean parameter".to_string()));

        let prop = ParameterProperty::from_number("A number parameter");
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_name, "number");
        assert_eq!(deserialized.description, Some("A number parameter".to_string()));
    }

    #[test]
    fn test_parameter_property_enum_serialization() {
        let prop = ParameterProperty::from_string("An enum parameter").add_enum_values(vec!["value1", "value2", "value3"]);
        let serialized = serde_json::to_string(&prop).unwrap();
        let deserialized = serde_json::from_str::<ParameterProperty>(&serialized).unwrap();
        assert_eq!(deserialized.type_name, "string");
        assert_eq!(deserialized.enum_values, Some(vec!["value1".to_string(), "value2".to_string(), "value3".to_string()]));
        assert_eq!(deserialized.description, Some("An enum parameter".to_string()));
    }
}
