//! # JSON Schema Builder
//!
//! This module provides functionality for building JSON schemas that can be used with
//! OpenAI's structured output features. JSON schemas define the expected structure
//! and format of data, allowing AI models to generate responses that conform to
//! specific formats.
//!
//! ## Features
//!
//! - **Type Safety**: Strongly typed schema construction
//! - **Flexible Properties**: Support for various JSON types (string, number, boolean, array, object)
//! - **Nested Structures**: Support for complex nested objects and arrays
//! - **Validation**: Built-in validation through required fields and type constraints
//! - **Serialization**: Direct serialization to JSON for API consumption
//!
//! ## Common Use Cases
//!
//! - **Data Extraction**: Extract structured information from unstructured text
//! - **Form Generation**: Generate forms with specific field requirements
//! - **API Responses**: Ensure consistent response formats
//! - **Configuration**: Define configuration schema for applications
//!
//! ## Example
//!
//! ```rust
//! use openai_tools::structured_output::Schema;
//!
//! // Create a schema for a person object
//! let mut schema = Schema::chat_json_schema("person".to_string());
//!
//! // Add basic properties
//! schema.add_property(
//!     "name".to_string(),
//!     "string".to_string(),
//!     Some("The person's full name".to_string())
//! );
//! schema.add_property(
//!     "age".to_string(),
//!     "number".to_string(),
//!     Some("The person's age in years".to_string())
//! );
//!
//! // Add an array property
//! schema.add_array(
//!     "hobbies".to_string(),
//!     vec![
//!         ("name".to_string(), "The name of the hobby".to_string()),
//!         ("level".to_string(), "Skill level (beginner, intermediate, advanced)".to_string()),
//!     ]
//! );
//!
//! // Convert to JSON for use with OpenAI API
//! let json_string = serde_json::to_string(&schema).unwrap();
//! println!("{}", json_string);
//! ```

use fxhash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Represents a single property or item type within a JSON schema.
///
/// This structure defines the type and characteristics of individual properties
/// in a JSON schema. It can represent simple types (string, number, boolean)
/// or complex types (arrays, objects) with nested structures.
///
/// # Fields
///
/// * `type_name` - The JSON type of this item (e.g., "string", "number", "array", "object")
/// * `description` - Optional human-readable description of the property
/// * `items` - For array types, defines the structure of array elements
///
/// # Supported Types
///
/// - **"string"**: Text values
/// - **"number"**: Numeric values (integers and floats)
/// - **"boolean"**: True/false values
/// - **"array"**: Lists of items with defined structure
/// - **"object"**: Complex nested objects
///
/// # Example
///
/// ```rust
/// use openai_tools::structured_output::ItemType;
///
/// // Simple string property
/// let name_prop = ItemType::new(
///     "string".to_string(),
///     Some("The user's full name".to_string())
/// );
///
/// // Numeric property
/// let age_prop = ItemType::new(
///     "number".to_string(),
///     Some("Age in years".to_string())
/// );
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ItemType {
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonItem>>,
}

impl ItemType {
    /// Creates a new `ItemType` with the specified type and optional description.
    ///
    /// # Arguments
    ///
    /// * `type_name` - The JSON type name (e.g., "string", "number", "boolean", "array", "object")
    /// * `description` - Optional description explaining the purpose of this property
    ///
    /// # Returns
    ///
    /// A new `ItemType` instance with the specified type and description.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::ItemType;
    ///
    /// // Create a string property with description
    /// let email = ItemType::new(
    ///     "string".to_string(),
    ///     Some("A valid email address".to_string())
    /// );
    ///
    /// // Create a number property without description
    /// let count = ItemType::new("number".to_string(), None);
    /// ```
    pub fn new(type_name: String, description: Option<String>) -> Self {
        Self {
            type_name: type_name.to_string(),
            description: description,
            items: None,
        }
    }

    /// Creates a deep clone of this `ItemType` instance.
    ///
    /// This method performs a deep copy of all nested structures, including
    /// any complex `items` that may be present for array types.
    ///
    /// # Returns
    ///
    /// A new `ItemType` instance that is an exact copy of this one.
    ///
    /// # Note
    ///
    /// This method is more explicit than the auto-derived `Clone` trait
    /// and ensures proper deep copying of nested Box<JsonItem> structures.
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
            items: if self.items.is_some() {
                Option::from(Box::new(items))
            } else {
                None
            },
        }
    }
}

/// Represents a JSON object structure with properties, requirements, and constraints.
///
/// This structure defines a JSON object schema including its properties, which fields
/// are required, and whether additional properties are allowed. It's used to build
/// complex nested object structures within JSON schemas.
///
/// # Fields
///
/// * `type_name` - Always "object" for object schemas
/// * `properties` - Map of property names to their type definitions
/// * `required` - List of property names that must be present
/// * `additional_properties` - Whether properties not defined in schema are allowed
///
/// # Schema Validation
///
/// - **Required Fields**: Properties listed in `required` must be present in valid JSON
/// - **Type Validation**: Each property must match its defined type
/// - **Additional Properties**: When false, only defined properties are allowed
///
/// # Example
///
/// ```rust
/// use openai_tools::structured_output::{JsonItem, ItemType};
/// use fxhash::FxHashMap;
///
/// // Create properties map
/// let mut properties = FxHashMap::default();
/// properties.insert(
///     "name".to_string(),
///     ItemType::new("string".to_string(), Some("Person's name".to_string()))
/// );
/// properties.insert(
///     "age".to_string(),
///     ItemType::new("number".to_string(), Some("Person's age".to_string()))
/// );
///
/// // Create object schema
/// let person_schema = JsonItem::new("object", properties);
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
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
    /// Creates a new `JsonItem` object schema with the specified properties.
    ///
    /// All properties provided will automatically be marked as required.
    /// Additional properties are disabled by default for strict validation.
    ///
    /// # Arguments
    ///
    /// * `type_name` - The type name (typically "object")
    /// * `properties` - Map of property names to their type definitions
    ///
    /// # Returns
    ///
    /// A new `JsonItem` with all provided properties marked as required.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::{JsonItem, ItemType};
    /// use fxhash::FxHashMap;
    ///
    /// let mut props = FxHashMap::default();
    /// props.insert("id".to_string(), ItemType::new("number".to_string(), None));
    /// props.insert("title".to_string(), ItemType::new("string".to_string(), None));
    ///
    /// let schema = JsonItem::new("object", props);
    /// // Both "id" and "title" will be required
    /// ```
    pub fn new(type_name: &str, properties: FxHashMap<String, ItemType>) -> Self {
        let mut required = Vec::new();
        for key in properties.keys() {
            required.push(key.clone());
        }
        Self {
            type_name: type_name.to_string(),
            properties,
            required: if required.is_empty() {
                None
            } else {
                Option::from(required)
            },
            additional_properties: false,
        }
    }

    /// Creates a default empty object schema.
    ///
    /// Returns a new `JsonItem` representing an empty object with no properties,
    /// no required fields, and additional properties disabled.
    ///
    /// # Returns
    ///
    /// A new empty `JsonItem` ready for property addition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::{JsonItem, ItemType};
    ///
    /// let mut schema = JsonItem::default();
    /// // Add properties later using add_property()
    /// ```
    pub fn default() -> Self {
        Self {
            type_name: "object".to_string(),
            properties: FxHashMap::default(),
            required: None,
            additional_properties: false,
        }
    }

    /// Adds a property to this object schema and marks it as required.
    ///
    /// This method adds a new property to the schema and automatically
    /// updates the required fields list to include this property.
    ///
    /// # Arguments
    ///
    /// * `prop_name` - The name of the property to add
    /// * `item` - The type definition for this property
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::{JsonItem, ItemType};
    ///
    /// let mut schema = JsonItem::default();
    /// let string_prop = ItemType::new("string".to_string(), Some("A name".to_string()));
    ///
    /// schema.add_property("name".to_string(), string_prop);
    /// // "name" is now a required property
    /// ```
    pub fn add_property(&mut self, prop_name: String, item: ItemType) {
        self.properties.insert(prop_name.to_string(), item.clone());
        if self.required.is_none() {
            self.required = Option::from(vec![prop_name.to_string()]);
        } else {
            let mut required = self.required.clone().unwrap();
            required.push(prop_name.to_string());
            self.required = Option::from(required);
        }
    }

    /// Adds an array property with the specified item structure.
    ///
    /// This method creates an array property where each element conforms
    /// to the provided `JsonItem` structure. The array property is automatically
    /// marked as required.
    ///
    /// # Arguments
    ///
    /// * `prop_name` - The name of the array property
    /// * `items` - The schema definition for array elements
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::{JsonItem, ItemType};
    /// use fxhash::FxHashMap;
    ///
    /// let mut schema = JsonItem::default();
    ///
    /// // Define structure for array items
    /// let mut item_props = FxHashMap::default();
    /// item_props.insert("id".to_string(), ItemType::new("number".to_string(), None));
    /// item_props.insert("name".to_string(), ItemType::new("string".to_string(), None));
    /// let item_schema = JsonItem::new("object", item_props);
    ///
    /// schema.add_array("items".to_string(), item_schema);
    /// ```
    pub fn add_array(&mut self, prop_name: String, items: JsonItem) {
        let mut prop = ItemType::new(String::from("array"), None);
        prop.items = Option::from(Box::new(items));
        self.properties.insert(prop_name.to_string(), prop);
        self.required = Option::from(vec![prop_name.to_string()]);
    }

    /// Creates a deep clone of this `JsonItem` instance.
    ///
    /// This method performs a deep copy of all properties and nested structures,
    /// ensuring that modifications to the clone don't affect the original.
    ///
    /// # Returns
    ///
    /// A new `JsonItem` instance that is an exact copy of this one.
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

/// Complete JSON schema definition with name and structure.
///
/// This is the top-level structure for defining JSON schemas that can be used
/// with OpenAI's structured output features. It combines a schema name with
/// the actual schema definition to create a complete, reusable schema.
///
/// # Fields
///
/// * `name` - A unique identifier for this schema
/// * `schema` - The actual schema definition describing the structure
///
/// # Usage with OpenAI API
///
/// This structure can be directly serialized and used with OpenAI's chat completion
/// API to ensure structured responses that conform to the defined schema.
///
/// # Example
///
/// ```rust
/// use openai_tools::structured_output::Schema;
///
/// // Create a schema for extracting contact information
/// let mut contact_schema = Schema::chat_json_schema("contact_info".to_string());
///
/// contact_schema.add_property(
///     "name".to_string(),
///     "string".to_string(),
///     Some("Full name of the person".to_string())
/// );
/// contact_schema.add_property(
///     "email".to_string(),
///     "string".to_string(),
///     Some("Email address".to_string())
/// );
/// contact_schema.add_property(
///     "phone".to_string(),
///     "string".to_string(),
///     Some("Phone number".to_string())
/// );
///
/// // Serialize for API use
/// let json_string = serde_json::to_string(&contact_schema).unwrap();
/// ```
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Schema {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    type_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<JsonItem>,
}

impl Schema {
    /// Creates a new `JsonSchema` with the specified name.
    ///
    /// This creates an empty object schema that can be populated with properties
    /// using the various `add_*` methods.
    ///
    /// # Arguments
    ///
    /// * `name` - A unique identifier for this schema
    ///
    /// # Returns
    ///
    /// A new `JsonSchema` instance with an empty object schema.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::Schema;
    ///
    /// let schema = Schema::chat_json_schema("user_profile".to_string());
    /// // Schema is now ready for property addition
    /// ```
    pub fn responses_text_schema() -> Self {
        Self {
            type_name: Some("text".to_string()),
            name: None,
            schema: None,
        }
    }

    /// Creates a new `JsonSchema` with the specified name (alternative constructor).
    ///
    /// This method is functionally identical to `new()` but provides an alternative
    /// naming convention for schema creation.
    ///
    /// # Arguments
    ///
    /// * `name` - A unique identifier for this schema
    ///
    /// # Returns
    ///
    /// A new `JsonSchema` instance with an empty object schema.
    pub fn responses_json_schema(name: String) -> Self {
        Self {
            type_name: Some("json_schema".to_string()),
            name: Some(name.to_string()),
            schema: Some(JsonItem::default()),
        }
    }

    /// Creates a new `JsonSchema` for chat completions with the specified name.
    ///
    /// This method creates a schema specifically designed for use with OpenAI's chat
    /// completion API. Unlike `responses_json_schema()`, this method doesn't set a
    /// type name, making it suitable for chat-based structured outputs.
    ///
    /// # Arguments
    ///
    /// * `name` - A unique identifier for this schema
    ///
    /// # Returns
    ///
    /// A new `JsonSchema` instance with an empty object schema optimized for chat completions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::Schema;
    ///
    /// let mut chat_schema = Schema::chat_json_schema("chat_response".to_string());
    /// chat_schema.add_property(
    ///     "response".to_string(),
    ///     "string".to_string(),
    ///     Some("The AI's response message".to_string())
    /// );
    /// ```
    pub fn chat_json_schema(name: String) -> Self {
        Self {
            type_name: None,
            name: Some(name.to_string()),
            schema: Some(JsonItem::default()),
        }
    }

    /// Adds a simple property to the schema.
    ///
    /// This method adds a property with a basic type (string, number, boolean, etc.)
    /// to the schema. The property is automatically marked as required.
    ///
    /// # Arguments
    ///
    /// * `prop_name` - The name of the property
    /// * `type_name` - The JSON type (e.g., "string", "number", "boolean")
    /// * `description` - Optional description explaining the property's purpose
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::Schema;
    ///
    /// let mut schema = Schema::chat_json_schema("person".to_string());
    ///
    /// // Add string property with description
    /// schema.add_property(
    ///     "name".to_string(),
    ///     "string".to_string(),
    ///     Some("The person's full name".to_string())
    /// );
    ///
    /// // Add number property without description
    /// schema.add_property(
    ///     "age".to_string(),
    ///     "number".to_string(),
    ///     None
    /// );
    ///
    /// // Add boolean property
    /// schema.add_property(
    ///     "is_active".to_string(),
    ///     "boolean".to_string(),
    ///     Some("Whether the person is currently active".to_string())
    /// );
    /// ```
    pub fn add_property(
        &mut self,
        prop_name: String,
        type_name: String,
        description: Option<String>,
    ) {
        let new_item = ItemType::new(type_name, description);
        self.schema
            .as_mut()
            .unwrap()
            .add_property(prop_name, new_item);
    }

    /// Adds an array property with string elements to the schema.
    ///
    /// This method creates an array property where each element is an object
    /// with string properties. All specified properties in the array elements
    /// are marked as required.
    ///
    /// # Arguments
    ///
    /// * `prop_name` - The name of the array property
    /// * `items` - Vector of (property_name, description) tuples for array elements
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::Schema;
    ///
    /// let mut schema = Schema::chat_json_schema("user_profile".to_string());
    ///
    /// // Add an array of address objects
    /// schema.add_array(
    ///     "addresses".to_string(),
    ///     vec![
    ///         ("street".to_string(), "Street address".to_string()),
    ///         ("city".to_string(), "City name".to_string()),
    ///         ("state".to_string(), "State or province".to_string()),
    ///         ("zip_code".to_string(), "Postal code".to_string()),
    ///     ]
    /// );
    ///
    /// // This creates an array where each element must have all four properties
    /// ```
    ///
    /// # Note
    ///
    /// Currently, this method only supports arrays of objects with string properties.
    /// For more complex array structures, you may need to manually construct the
    /// schema using `JsonItem` and `ItemType`.
    pub fn add_array(&mut self, prop_name: String, items: Vec<(String, String)>) {
        let mut array_item = JsonItem::default();
        for (name, description) in items.iter() {
            let item = ItemType::new(String::from("string"), Option::from(description.clone()));
            array_item.add_property(name.clone(), item);
        }
        self.schema
            .as_mut()
            .unwrap()
            .add_array(prop_name, array_item);
    }

    /// Creates a deep clone of this `JsonSchema` instance.
    ///
    /// This method performs a deep copy of the entire schema structure,
    /// including all nested properties and their definitions.
    ///
    /// # Returns
    ///
    /// A new `JsonSchema` instance that is an exact copy of this one.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::structured_output::Schema;
    ///
    /// let mut original = Schema::chat_json_schema("template".to_string());
    /// original.add_property("id".to_string(), "number".to_string(), None);
    ///
    /// let mut copy = original.clone();
    /// copy.add_property("name".to_string(), "string".to_string(), None);
    ///
    /// // original still only has "id", copy has both "id" and "name"
    /// ```
    pub fn clone(&self) -> Self {
        Self {
            type_name: self.type_name.clone(),
            name: self.name.clone(),
            schema: self.schema.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_build_json_schema_simple() {
        let mut json_schema = Schema::chat_json_schema(String::from("test-schema"));
        json_schema.add_property(String::from("test_property"), String::from("string"), None);

        let schema_string = serde_json::to_string(&json_schema).unwrap();
        println!("{}", serde_json::to_string_pretty(&json_schema).unwrap());

        assert_eq!(
            schema_string,
            r#"{"name":"test-schema","schema":{"type":"object","properties":{"test_property":{"type":"string"}},"required":["test_property"],"additionalProperties":false}}"#
        );
    }

    #[tokio::test]
    async fn test_build_json_schema_with_description() {
        let mut json_schema = Schema::chat_json_schema(String::from("test-schema"));
        json_schema.add_property(
            String::from("email"),
            String::from("string"),
            Some(String::from("The email address that appears in the input")),
        );

        let schema_string = serde_json::to_string(&json_schema).unwrap();
        println!("{}", serde_json::to_string_pretty(&json_schema).unwrap());

        assert_eq!(
            schema_string,
            r#"{"name":"test-schema","schema":{"type":"object","properties":{"email":{"type":"string","description":"The email address that appears in the input"}},"required":["email"],"additionalProperties":false}}"#,
        );
    }

    #[tokio::test]
    async fn test_build_json_schema_add_array() {
        let mut json_schema = Schema::chat_json_schema(String::from("test-schema"));
        json_schema.add_property(
            String::from("test-property"),
            String::from("string"),
            Some(String::from("This is a test property")),
        );
        json_schema.add_array(
            String::from("test-array"),
            vec![
                (
                    String::from("test-array-property-1"),
                    String::from("This is test array property 1."),
                ),
                (
                    String::from("test-array-property-2"),
                    String::from("This is test array property 2."),
                ),
            ],
        );
        let schema_string = serde_json::to_string(&json_schema).unwrap();

        println!("{}", serde_json::to_string_pretty(&json_schema).unwrap());

        assert_eq!(
            schema_string,
            r#"{"name":"test-schema","schema":{"type":"object","properties":{"test-property":{"type":"string","description":"This is a test property"},"test-array":{"type":"array","items":{"type":"object","properties":{"test-array-property-2":{"type":"string","description":"This is test array property 2."},"test-array-property-1":{"type":"string","description":"This is test array property 1."}},"required":["test-array-property-1","test-array-property-2"],"additionalProperties":false}}},"required":["test-array"],"additionalProperties":false}}"#
        );
    }
}
