use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ItemType {
    #[serde(rename = "type")]
    type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    items: Option<Box<JsonItem>>,
}

impl ItemType {
    pub fn new<T: AsRef<str>, U: AsRef<str>>(type_name: T, description: U) -> Self {
        Self {
            type_name: type_name.as_ref().to_string(),
            description: match description.as_ref() {
                "" => None,
                _ => Some(description.as_ref().to_string()),
            },
            items: None,
        }
    }

    pub fn clone(&self) -> Self {
        let mut items: JsonItem = JsonItem::default();
        if let Some(item) = &self.items {
            let mut _properties: HashMap<String, ItemType> = HashMap::default();
            for (key, value) in item.properties.iter() {
                _properties.insert(key.clone(), value.clone());
            }
            items.type_name = item.type_name.clone();
            items.properties = _properties;
            items.required = item.required.clone();
            items.additional_properties = item.additional_properties;
        }

        Self {
            type_name: self.type_name.clone(),
            description: self.description.clone(),
            items: if self.items.is_some() { Option::from(Box::new(items)) } else { None },
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct JsonItem {
    #[serde(rename = "type")]
    type_name: Option<String>,
    properties: HashMap<String, ItemType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required: Option<Vec<String>>,
    #[serde(rename = "additionalProperties")]
    additional_properties: bool,
}

impl JsonItem {
    fn add_property<T: AsRef<str>>(&mut self, prop_name: T, item: ItemType) {
        self.properties.insert(prop_name.as_ref().to_string(), item.clone());
        if self.required.is_none() {
            self.required = Some(vec![]);
        }
        self.required.as_mut().unwrap().push(prop_name.as_ref().to_string());
    }

    fn add_array<T: AsRef<str>>(&mut self, prop_name: T, items: JsonItem) {
        let mut prop = ItemType::new("array", "");
        prop.items = Option::from(Box::new(items));
        self.properties.insert(prop_name.as_ref().to_string(), prop);
        if self.required.is_none() {
            self.required = Some(vec![]);
        }
        self.required.as_mut().unwrap().push(prop_name.as_ref().to_string());
    }
}

impl Default for JsonItem {
    fn default() -> Self {
        Self { type_name: Some("object".to_string()), properties: HashMap::new(), required: None, additional_properties: false }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Schema {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    type_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<JsonItem>,
}

impl Schema {
    pub fn responses_text_schema() -> Self {
        Self { type_name: Some("text".to_string()), name: None, schema: None }
    }

    pub fn responses_json_schema<T: AsRef<str>>(name: T) -> Self {
        Self { type_name: Some("json_schema".to_string()), name: Some(name.as_ref().to_string()), schema: Some(JsonItem::default()) }
    }

    pub fn chat_json_schema<T: AsRef<str>>(name: T) -> Self {
        Self { type_name: None, name: Some(name.as_ref().to_string()), schema: Some(JsonItem::default()) }
    }

    pub fn add_property<T: AsRef<str>, U: AsRef<str>, V: AsRef<str>>(&mut self, prop_name: T, type_name: U, description: V) {
        let new_item = ItemType::new(type_name, description);
        self.schema.as_mut().unwrap().add_property(prop_name, new_item);
    }

    pub fn add_array<T: AsRef<str>, U: AsRef<str>>(&mut self, prop_name: T, items: Vec<(U, U)>) {
        let mut array_item = JsonItem::default();
        for (name, description) in items.iter() {
            let item = ItemType::new("string", description.as_ref());
            array_item.add_property(name.as_ref(), item);
        }
        self.schema.as_mut().unwrap().add_array(prop_name, array_item);
    }
}
