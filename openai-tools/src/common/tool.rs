use crate::common::{
    function::Function,
    parameters::{Name, ParameterProp, Parameters},
};
use serde::{Deserialize, Serialize};

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
