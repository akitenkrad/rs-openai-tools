//! Function module for OpenAI tools.
//!
//! This module provides the `Function` struct and related functionality for representing
//! and working with OpenAI function calls. Functions are used to define callable operations
//! that can be invoked by OpenAI's API, including their parameters, descriptions, and
//! execution modes.
//!
//! The `Function` struct is a core component that is used throughout the OpenAI tools library:
//!
//! - In [`crate::common::message`] - Used within `ToolCall` structures to represent function calls in messages
//! - In [`crate::common::tool`] - Used within `Tool` structures to define available functions for OpenAI models
//! - In [`crate::chat::request`] - Used in Chat Completion API requests for function calling capabilities
//! - In [`crate::responses::request`] - Used in Responses API requests for structured interactions
//!
//! ## Key Features
//!
//! - **Serialization/Deserialization**: Custom implementations for clean JSON output
//! - **Flexible Parameters**: Support for complex parameter schemas via [`crate::common::parameters::Parameters`]
//! - **Argument Handling**: Support for both JSON string and structured argument formats
//! - **Strict Mode**: Optional strict execution mode for enhanced validation
//!
//! ## Usage Patterns
//!
//! Functions are typically used in two main contexts:
//!
//! 1. **Tool Definition**: When creating tools that OpenAI models can call
//! 2. **Function Execution**: When processing function calls from OpenAI responses

use crate::common::{
    errors::{OpenAIToolError, Result as OpenAIToolResult},
    parameters::Parameters,
};
use serde::{ser::SerializeStruct, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Represents a function that can be called by OpenAI tools.
///
/// This structure contains metadata about a function including its name, description,
/// parameters, and whether it should be executed in strict mode.
#[derive(Debug, Clone, Default)]
pub struct Function {
    /// The name of the function
    pub name: String,
    /// Optional description of what the function does
    pub description: Option<String>,
    /// Optional parameters that the function accepts
    pub parameters: Option<Parameters>,
    /// Optional arguments passed to the function as key-value pairs
    pub arguments: Option<HashMap<String, Value>>,
    /// Whether the function should be executed in strict mode
    pub strict: bool,
}

impl Function {
    /// Creates a new Function instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the function
    /// * `description` - A description of what the function does
    /// * `parameters` - The parameters that the function accepts
    /// * `strict` - Whether the function should be executed in strict mode
    ///
    /// # Returns
    ///
    /// A new Function instance
    pub fn new<T: AsRef<str>, U: AsRef<str>>(name: T, description: U, parameters: Parameters, strict: bool) -> Self {
        Self {
            name: name.as_ref().to_string(),
            description: Some(description.as_ref().to_string()),
            parameters: Some(parameters),
            strict,
            ..Default::default()
        }
    }

    /// Returns the function arguments as a HashMap.
    ///
    /// # Returns
    ///
    /// * `Ok(HashMap<String, Value>)` - The arguments as a map if they exist
    /// * `Err(OpenAIToolError)` - If the arguments are not set
    ///
    /// # Errors
    ///
    /// This function will return an error if the arguments are not set.
    pub fn arguments_as_map(&self) -> OpenAIToolResult<HashMap<String, Value>> {
        if let Some(args) = &self.arguments {
            Ok(args.clone())
        } else {
            Err(OpenAIToolError::from(anyhow::anyhow!("Function arguments are not set")))
        }
    }
}

/// Custom serialization implementation for Function.
///
/// This implementation ensures that only non-None optional fields are serialized,
/// keeping the JSON output clean and avoiding null values for optional fields.
impl Serialize for Function {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Function", 4)?;
        state.serialize_field("name", &self.name)?;
        if let Some(description) = &self.description {
            state.serialize_field("description", description)?;
        }
        if let Some(parameters) = &self.parameters {
            state.serialize_field("parameters", parameters)?;
        }
        if let Some(arguments) = &self.arguments {
            state.serialize_field("arguments", arguments)?;
        }
        state.serialize_field("strict", &self.strict)?;

        if let Some(arguments) = &self.arguments {
            if !arguments.is_empty() {
                state.serialize_field("arguments", &serde_json::to_string(arguments).expect("Failed to serialize arguments in Function"))?;
            }
        }
        state.end()
    }
}

/// Custom deserialization implementation for Function.
///
/// This implementation handles the deserialization of Function from JSON,
/// parsing arguments as JSON strings and converting parameters from JSON objects.
/// The `name` field is required, while other fields are optional.
impl<'de> Deserialize<'de> for Function {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut function = Function::default();
        let map: HashMap<String, Value> = HashMap::deserialize(deserializer)?;

        if let Some(name) = map.get("name").and_then(Value::as_str) {
            function.name = name.to_string();
        } else {
            return Err(serde::de::Error::missing_field("name"));
        }

        let arguments = map.get("arguments").and_then(Value::as_str);
        if let Some(args) = arguments {
            function.arguments = serde_json::from_str(args).ok();
        } else {
            function.arguments = None;
        }

        let parameters = map.get("parameters").and_then(Value::as_object);
        if let Some(params) = parameters {
            function.parameters = Some(Parameters::deserialize(Value::Object(params.clone())).map_err(serde::de::Error::custom)?);
        } else {
            function.parameters = None;
        }

        function.description = map.get("description").and_then(Value::as_str).map(String::from);
        function.strict = map.get("strict").and_then(Value::as_bool).unwrap_or(false);

        Ok(function)
    }
}
