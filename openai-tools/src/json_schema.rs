use fxhash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct ItemType {
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonItem>>,
}

impl ItemType {
    pub fn new(type_name: String, description: Option<String>) -> Self {
        Self {
            type_name,
            description: description,
            items: None,
        }
    }

    pub fn clone(&self) -> Self {
        let mut items: JsonItem = JsonItem::default();
        if let Some(item) = &self.items {
            let mut _properties: FxHashMap<String, ItemType> = FxHashMap::default();
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

#[derive(Debug, Deserialize, Serialize)]
pub struct JsonItem {
    #[serde(rename = "type")]
    pub type_name: String,
    pub properties: FxHashMap<String, ItemType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(rename = "additionalProperties")]
    pub additional_properties: bool,
}

impl JsonItem {
    pub fn new(type_name: String, properties: FxHashMap<String, ItemType>) -> Self {
        let mut required = Vec::new();
        for key in properties.keys() {
            required.push(key.clone());
        }
        Self {
            type_name,
            properties,
            required: if required.is_empty() { None } else { Option::from(required) },
            additional_properties: false,
        }
    }

    pub fn default() -> Self {
        Self {
            type_name: "object".to_string(),
            properties: FxHashMap::default(),
            required: None,
            additional_properties: false,
        }
    }

    pub fn add_property(&mut self, prop_name: String, item: ItemType) {
        self.properties.insert(prop_name.clone(), item.clone());
        if self.required.is_none() {
            self.required = Option::from(vec![prop_name.clone()]);
        } else {
            let mut required = self.required.clone().unwrap();
            required.push(prop_name.clone());
            self.required = Option::from(required);
        }
    }

    pub fn add_array(&mut self, prop_name: String, items: JsonItem) {
        let mut prop = ItemType::new("array".to_string(), None);
        prop.items = Option::from(Box::new(items));
        self.properties.insert(prop_name.clone(), prop);
        self.required = Option::from(vec![prop_name.clone()]);
    }

    pub fn clone(&self) -> Self {
        let mut properties: FxHashMap<String, ItemType> = FxHashMap::default();
        for (key, value) in self.properties.iter() {
            properties.insert(key.clone(), value.clone());
        }
        Self {
            type_name: self.type_name.clone(),
            properties: properties,
            required: self.required.clone(),
            additional_properties: self.additional_properties,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct JsonSchema {
    pub name: String,
    pub schema: JsonItem,
}

impl JsonSchema {
    pub fn new(name: String) -> Self {
        let schema = JsonItem::default();
        Self { name, schema }
    }

    pub fn new_schema(name: String) -> Self {
        Self {
            name,
            schema: JsonItem::default(),
        }
    }

    pub fn add_property(&mut self, prop_name: String, type_name: String, description: Option<String>) {
        let new_item = ItemType::new(type_name, description);
        self.schema.add_property(prop_name, new_item);
    }

    pub fn add_array(&mut self, prop_name: String, items: Vec<(String, String)>) {
        let mut array_item = JsonItem::default();
        for (name, description) in items.iter() {
            let item = ItemType::new("string".to_string(), Option::from(description.clone()));
            array_item.add_property(name.clone(), item);
        }
        self.schema.add_array(prop_name, array_item);
    }

    pub fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            schema: self.schema.clone(),
        }
    }
}
