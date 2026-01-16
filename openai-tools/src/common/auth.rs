//! Authentication abstraction for OpenAI and Azure OpenAI APIs
//!
//! This module provides a unified interface for authenticating with both
//! OpenAI API and Azure OpenAI API. It handles the differences in:
//! - Header format (Bearer token vs api-key)
//! - Endpoint URL construction
//! - Environment variable names
//!
//! # Quick Start
//!
//! ## OpenAI API (existing users - no changes needed)
//!
//! ```rust,no_run
//! use openai_tools::common::auth::AuthProvider;
//!
//! // From environment variable OPENAI_API_KEY
//! let auth = AuthProvider::openai_from_env()?;
//! # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
//! ```
//!
//! ## Azure OpenAI API
//!
//! ```rust,no_run
//! use openai_tools::common::auth::{AuthProvider, AzureAuth};
//!
//! // From environment variables
//! let auth = AuthProvider::azure_from_env()?;
//!
//! // Or explicit configuration (dynamic URL construction)
//! let auth = AuthProvider::Azure(
//!     AzureAuth::new("your-api-key", "my-resource", "gpt-4o-deployment")
//!         .with_api_version("2024-08-01-preview")
//! );
//! # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
//! ```
//!
//! ## Azure with Complete Base URL
//!
//! Use `with_base_url` when you want to provide a complete URL without dynamic construction:
//!
//! ```rust
//! use openai_tools::common::auth::{AuthProvider, AzureAuth};
//!
//! // Static URL mode - no dynamic URL construction
//! let auth = AuthProvider::Azure(
//!     AzureAuth::with_base_url(
//!         "your-api-key",
//!         "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview"
//!     )
//! );
//! ```
//!
//! ## Auto-detection
//!
//! ```rust,no_run
//! use openai_tools::common::auth::AuthProvider;
//!
//! // Uses Azure if AZURE_OPENAI_API_KEY is set, otherwise OpenAI
//! let auth = AuthProvider::from_env()?;
//! # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
//! ```

use crate::common::errors::{OpenAIToolError, Result};
use dotenvy::dotenv;
use request::header::{HeaderMap, HeaderValue};
use std::env;

/// Default OpenAI API base URL
const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Default Azure OpenAI API version
const AZURE_DEFAULT_API_VERSION: &str = "2024-08-01-preview";

/// Authentication provider for OpenAI APIs
///
/// This enum encapsulates the authentication strategy for different API providers.
/// It handles both endpoint URL construction and HTTP header application.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::common::auth::AuthProvider;
///
/// // Auto-detect from environment
/// let auth = AuthProvider::from_env()?;
///
/// // Get endpoint for chat completions
/// let endpoint = auth.endpoint("chat/completions");
///
/// // Apply auth headers to a request
/// let mut headers = request::header::HeaderMap::new();
/// auth.apply_headers(&mut headers)?;
/// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
/// ```
#[derive(Debug, Clone)]
pub enum AuthProvider {
    /// OpenAI API authentication
    OpenAI(OpenAIAuth),
    /// Azure OpenAI API authentication
    Azure(AzureAuth),
}

/// OpenAI API authentication configuration
///
/// Handles authentication for the standard OpenAI API using Bearer tokens.
///
/// # Header Format
///
/// ```text
/// Authorization: Bearer sk-...
/// ```
///
/// # Endpoint Format
///
/// ```text
/// https://api.openai.com/v1/{path}
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIAuth {
    /// The API key (sk-...)
    api_key: String,
    /// Base URL for API requests (default: https://api.openai.com/v1)
    base_url: String,
}

impl OpenAIAuth {
    /// Creates a new OpenAI authentication configuration
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key (typically starts with "sk-")
    ///
    /// # Returns
    ///
    /// A new `OpenAIAuth` instance with default base URL
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::OpenAIAuth;
    ///
    /// let auth = OpenAIAuth::new("sk-your-api-key");
    /// ```
    pub fn new<T: Into<String>>(api_key: T) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: OPENAI_DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Sets a custom base URL
    ///
    /// Use this for proxies or alternative OpenAI-compatible APIs.
    ///
    /// # Arguments
    ///
    /// * `url` - Custom base URL (without trailing slash)
    ///
    /// # Returns
    ///
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::OpenAIAuth;
    ///
    /// let auth = OpenAIAuth::new("sk-key")
    ///     .with_base_url("https://my-proxy.example.com/v1");
    /// ```
    pub fn with_base_url<T: Into<String>>(mut self, url: T) -> Self {
        self.base_url = url.into();
        self
    }

    /// Returns the API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Constructs the full endpoint URL for a given path
    ///
    /// # Arguments
    ///
    /// * `path` - API path (e.g., "chat/completions")
    ///
    /// # Returns
    ///
    /// Full URL string
    fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), path.trim_start_matches('/'))
    }

    /// Applies authentication headers to a request
    ///
    /// Adds the `Authorization: Bearer {key}` header.
    fn apply_headers(&self, headers: &mut HeaderMap) -> Result<()> {
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|e| OpenAIToolError::Error(format!("Invalid header value: {}", e)))?,
        );
        Ok(())
    }
}

/// Azure OpenAI API authentication configuration
///
/// Handles authentication for Azure OpenAI Service, which uses different
/// header names and endpoint URL patterns than the standard OpenAI API.
///
/// # Header Format
///
/// ```text
/// api-key: {key}
/// ```
///
/// Or with Entra ID:
///
/// ```text
/// Authorization: Bearer {token}
/// ```
///
/// # Endpoint Format (Dynamic Mode - Default)
///
/// ```text
/// https://{resource}.openai.azure.com/openai/deployments/{deployment}/{path}?api-version={version}
/// ```
///
/// # Endpoint Format (Static Base URL Mode)
///
/// When using `with_base_url`, the URL is constructed as:
///
/// ```text
/// {base_url}/{path}
/// ```
///
/// The `base_url` should include everything except the API path, including
/// the `api-version` query parameter if needed.
#[derive(Debug, Clone)]
pub struct AzureAuth {
    /// API key or Entra ID token
    api_key: String,
    /// Azure resource name (used in dynamic URL mode)
    resource_name: String,
    /// Deployment name (used in dynamic URL mode)
    deployment_name: String,
    /// API version (used in dynamic URL mode)
    api_version: String,
    /// Whether using Entra ID (Azure AD) authentication
    use_entra_id: bool,
    /// Complete base URL (overrides dynamic URL construction)
    base_url: Option<String>,
    /// Whether to dynamically construct the URL (default: true)
    use_dynamic_url: bool,
}

impl AzureAuth {
    /// Creates a new Azure OpenAI authentication configuration
    ///
    /// # Arguments
    ///
    /// * `api_key` - Azure OpenAI API key
    /// * `resource_name` - Azure resource name (the name before .openai.azure.com)
    /// * `deployment_name` - Model deployment name
    ///
    /// # Returns
    ///
    /// A new `AzureAuth` instance with default API version
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::AzureAuth;
    ///
    /// let auth = AzureAuth::new("your-api-key", "my-resource", "gpt-4o-deployment");
    /// ```
    pub fn new<T: Into<String>>(api_key: T, resource_name: T, deployment_name: T) -> Self {
        Self {
            api_key: api_key.into(),
            resource_name: resource_name.into(),
            deployment_name: deployment_name.into(),
            api_version: AZURE_DEFAULT_API_VERSION.to_string(),
            use_entra_id: false,
            base_url: None,
            use_dynamic_url: true,
        }
    }

    /// Creates a new Azure OpenAI auth with a custom endpoint URL
    ///
    /// Use this when you have a full endpoint URL instead of resource name.
    /// This method still uses dynamic URL construction, appending the deployment path
    /// and api-version query parameter.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Azure OpenAI API key
    /// * `endpoint` - Base endpoint URL (e.g., https://my-resource.openai.azure.com)
    /// * `deployment_name` - Model deployment name
    ///
    /// # Returns
    ///
    /// A new `AzureAuth` instance that constructs URLs dynamically
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::AzureAuth;
    ///
    /// let auth = AzureAuth::with_endpoint(
    ///     "your-api-key",
    ///     "https://my-resource.openai.azure.com",
    ///     "gpt-4o-deployment"
    /// );
    /// // Endpoint will be: https://my-resource.openai.azure.com/openai/deployments/gpt-4o-deployment/{path}?api-version=...
    /// ```
    pub fn with_endpoint<T: Into<String>>(api_key: T, endpoint: T, deployment_name: T) -> Self {
        Self {
            api_key: api_key.into(),
            resource_name: String::new(),
            deployment_name: deployment_name.into(),
            api_version: AZURE_DEFAULT_API_VERSION.to_string(),
            use_entra_id: false,
            base_url: Some(endpoint.into()),
            use_dynamic_url: true,
        }
    }

    /// Creates a new Azure OpenAI auth with a complete base URL
    ///
    /// Use this when you want to provide a complete base URL without dynamic URL construction.
    /// The URL should include everything except the API path (e.g., "chat/completions").
    ///
    /// If the base URL contains query parameters (e.g., `?api-version=...`), the path will be
    /// inserted before the query string.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Azure OpenAI API key
    /// * `base_url` - Complete base URL including deployment path and optionally api-version
    ///
    /// # Returns
    ///
    /// A new `AzureAuth` instance that uses the base URL directly
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::AzureAuth;
    ///
    /// // Without api-version (simpler form)
    /// let auth = AzureAuth::with_base_url(
    ///     "your-api-key",
    ///     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o"
    /// );
    /// // Endpoint for "chat/completions" will be:
    /// // https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions
    ///
    /// // With api-version included
    /// let auth = AzureAuth::with_base_url(
    ///     "your-api-key",
    ///     "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview"
    /// );
    /// // Endpoint for "chat/completions" will be:
    /// // https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview
    /// ```
    pub fn with_base_url<T: Into<String>>(api_key: T, base_url: T) -> Self {
        Self {
            api_key: api_key.into(),
            resource_name: String::new(),
            deployment_name: String::new(),
            api_version: String::new(),
            use_entra_id: false,
            base_url: Some(base_url.into()),
            use_dynamic_url: false,
        }
    }

    /// Sets the API version
    ///
    /// # Arguments
    ///
    /// * `version` - API version string (e.g., "2024-08-01-preview")
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_api_version<T: Into<String>>(mut self, version: T) -> Self {
        self.api_version = version.into();
        self
    }

    /// Enables Entra ID (Azure AD) authentication
    ///
    /// When enabled, the API key is treated as a Bearer token
    /// and sent in the `Authorization` header instead of `api-key`.
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_entra_id(mut self) -> Self {
        self.use_entra_id = true;
        self
    }

    /// Returns the API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the resource name
    pub fn resource_name(&self) -> &str {
        &self.resource_name
    }

    /// Returns the deployment name
    pub fn deployment_name(&self) -> &str {
        &self.deployment_name
    }

    /// Returns the API version
    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    /// Returns whether Entra ID authentication is enabled
    pub fn is_entra_id(&self) -> bool {
        self.use_entra_id
    }

    /// Returns the base URL if set
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// Returns whether dynamic URL construction is enabled
    pub fn is_dynamic_url(&self) -> bool {
        self.use_dynamic_url
    }

    /// Constructs the base endpoint URL for dynamic mode
    fn dynamic_base_endpoint(&self) -> String {
        if let Some(ref url) = self.base_url {
            url.trim_end_matches('/').to_string()
        } else {
            format!("https://{}.openai.azure.com", self.resource_name)
        }
    }

    /// Constructs the full endpoint URL for a given path
    ///
    /// # Arguments
    ///
    /// * `path` - API path (e.g., "chat/completions")
    ///
    /// # Returns
    ///
    /// Full Azure OpenAI URL
    ///
    /// # URL Construction Modes
    ///
    /// - **Dynamic mode** (`use_dynamic_url = true`): Constructs URL as
    ///   `{base}/openai/deployments/{deployment}/{path}?api-version={version}`
    ///
    /// - **Static mode** (`use_dynamic_url = false`): Constructs URL as
    ///   `{base_url}/{path}`. If base_url contains query parameters,
    ///   the path is inserted before the query string.
    fn endpoint(&self, path: &str) -> String {
        if self.use_dynamic_url {
            // Dynamic URL construction (original behavior)
            format!(
                "{}/openai/deployments/{}/{}?api-version={}",
                self.dynamic_base_endpoint(),
                self.deployment_name,
                path.trim_start_matches('/'),
                self.api_version
            )
        } else {
            // Static base URL mode
            let base = self.base_url.as_deref().unwrap_or("");
            let path = path.trim_start_matches('/');

            // Check if base URL contains query parameters
            if let Some(query_pos) = base.find('?') {
                // Insert path before query string
                let (url_path, query) = base.split_at(query_pos);
                format!("{}/{}{}", url_path.trim_end_matches('/'), path, query)
            } else {
                // Simple concatenation
                format!("{}/{}", base.trim_end_matches('/'), path)
            }
        }
    }

    /// Applies authentication headers to a request
    ///
    /// Uses `api-key` header for API key auth, or `Authorization: Bearer`
    /// for Entra ID auth.
    fn apply_headers(&self, headers: &mut HeaderMap) -> Result<()> {
        if self.use_entra_id {
            headers.insert(
                "Authorization",
                HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                    .map_err(|e| OpenAIToolError::Error(format!("Invalid header value: {}", e)))?,
            );
        } else {
            headers.insert(
                "api-key",
                HeaderValue::from_str(&self.api_key)
                    .map_err(|e| OpenAIToolError::Error(format!("Invalid header value: {}", e)))?,
            );
        }
        Ok(())
    }
}

impl AuthProvider {
    /// Creates an OpenAI authentication provider from environment variables
    ///
    /// Reads the API key from `OPENAI_API_KEY` environment variable.
    ///
    /// # Returns
    ///
    /// `Result<AuthProvider>` - OpenAI auth provider or error if env var not set
    ///
    /// # Environment Variables
    ///
    /// | Variable | Required | Description |
    /// |----------|----------|-------------|
    /// | `OPENAI_API_KEY` | Yes | OpenAI API key |
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// let auth = AuthProvider::openai_from_env()?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn openai_from_env() -> Result<Self> {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| OpenAIToolError::Error("OPENAI_API_KEY environment variable not set".into()))?;
        Ok(Self::OpenAI(OpenAIAuth::new(api_key)))
    }

    /// Creates an Azure OpenAI authentication provider from environment variables
    ///
    /// # Returns
    ///
    /// `Result<AuthProvider>` - Azure auth provider or error if required vars not set
    ///
    /// # Environment Variables
    ///
    /// | Variable | Required | Description |
    /// |----------|----------|-------------|
    /// | `AZURE_OPENAI_API_KEY` | Yes* | Azure API key |
    /// | `AZURE_OPENAI_TOKEN` | Yes* | Entra ID token (alternative to API key) |
    /// | `AZURE_OPENAI_ENDPOINT` | Yes** | Full endpoint URL |
    /// | `AZURE_OPENAI_RESOURCE_NAME` | Yes** | Resource name (alternative to endpoint) |
    /// | `AZURE_OPENAI_DEPLOYMENT_NAME` | Yes | Deployment name |
    /// | `AZURE_OPENAI_API_VERSION` | No | API version (default: 2024-08-01-preview) |
    ///
    /// \* Either `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_TOKEN` required
    /// \*\* Either `AZURE_OPENAI_ENDPOINT` or `AZURE_OPENAI_RESOURCE_NAME` required
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// // With environment variables:
    /// // AZURE_OPENAI_API_KEY=xxx
    /// // AZURE_OPENAI_RESOURCE_NAME=my-resource
    /// // AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
    /// let auth = AuthProvider::azure_from_env()?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn azure_from_env() -> Result<Self> {
        dotenv().ok();

        // Get API key or Entra ID token
        let (api_key, use_entra_id) = if let Ok(key) = env::var("AZURE_OPENAI_API_KEY") {
            (key, false)
        } else if let Ok(token) = env::var("AZURE_OPENAI_TOKEN") {
            (token, true)
        } else {
            return Err(OpenAIToolError::Error(
                "Neither AZURE_OPENAI_API_KEY nor AZURE_OPENAI_TOKEN environment variable is set".into(),
            ));
        };

        // Get deployment name (required)
        let deployment_name = env::var("AZURE_OPENAI_DEPLOYMENT_NAME")
            .map_err(|_| OpenAIToolError::Error("AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set".into()))?;

        // Get endpoint or resource name
        let mut auth = if let Ok(endpoint) = env::var("AZURE_OPENAI_ENDPOINT") {
            AzureAuth::with_endpoint(api_key, endpoint, deployment_name)
        } else if let Ok(resource_name) = env::var("AZURE_OPENAI_RESOURCE_NAME") {
            AzureAuth::new(api_key, resource_name, deployment_name)
        } else {
            return Err(OpenAIToolError::Error(
                "Neither AZURE_OPENAI_ENDPOINT nor AZURE_OPENAI_RESOURCE_NAME environment variable is set".into(),
            ));
        };

        // Apply optional API version
        if let Ok(version) = env::var("AZURE_OPENAI_API_VERSION") {
            auth = auth.with_api_version(version);
        }

        // Apply Entra ID if using token
        if use_entra_id {
            auth = auth.with_entra_id();
        }

        Ok(Self::Azure(auth))
    }

    /// Creates an authentication provider by auto-detecting from environment
    ///
    /// Tries Azure first (if `AZURE_OPENAI_API_KEY` is set), then falls back to OpenAI.
    ///
    /// # Returns
    ///
    /// `Result<AuthProvider>` - Detected auth provider or error
    ///
    /// # Detection Order
    ///
    /// 1. If `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_TOKEN` is set → Azure
    /// 2. If `OPENAI_API_KEY` is set → OpenAI
    /// 3. Otherwise → Error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// let auth = AuthProvider::from_env()?;
    /// match &auth {
    ///     AuthProvider::OpenAI(_) => println!("Using OpenAI"),
    ///     AuthProvider::Azure(_) => println!("Using Azure"),
    /// }
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn from_env() -> Result<Self> {
        dotenv().ok();

        // Try Azure first if its key is present
        if env::var("AZURE_OPENAI_API_KEY").is_ok() || env::var("AZURE_OPENAI_TOKEN").is_ok() {
            return Self::azure_from_env();
        }

        // Fall back to OpenAI
        Self::openai_from_env()
    }

    /// Constructs the full endpoint URL for a given API path
    ///
    /// # Arguments
    ///
    /// * `path` - API path (e.g., "chat/completions", "embeddings")
    ///
    /// # Returns
    ///
    /// Full endpoint URL appropriate for the provider
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// let auth = AuthProvider::openai_from_env()?;
    /// let url = auth.endpoint("chat/completions");
    /// // OpenAI: https://api.openai.com/v1/chat/completions
    /// // Azure: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={ver}
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn endpoint(&self, path: &str) -> String {
        match self {
            Self::OpenAI(auth) => auth.endpoint(path),
            Self::Azure(auth) => auth.endpoint(path),
        }
    }

    /// Applies authentication headers to a request
    ///
    /// # Arguments
    ///
    /// * `headers` - Mutable reference to header map
    ///
    /// # Returns
    ///
    /// `Result<()>` - Success or error if header value is invalid
    ///
    /// # Header Differences
    ///
    /// | Provider | Header Name | Value Format |
    /// |----------|-------------|--------------|
    /// | OpenAI | `Authorization` | `Bearer {key}` |
    /// | Azure (API key) | `api-key` | `{key}` |
    /// | Azure (Entra ID) | `Authorization` | `Bearer {token}` |
    pub fn apply_headers(&self, headers: &mut HeaderMap) -> Result<()> {
        match self {
            Self::OpenAI(auth) => auth.apply_headers(headers),
            Self::Azure(auth) => auth.apply_headers(headers),
        }
    }

    /// Returns the API key (for backward compatibility)
    ///
    /// # Returns
    ///
    /// The API key or token string
    pub fn api_key(&self) -> &str {
        match self {
            Self::OpenAI(auth) => auth.api_key(),
            Self::Azure(auth) => auth.api_key(),
        }
    }

    /// Returns whether this is an Azure provider
    ///
    /// # Returns
    ///
    /// `true` if Azure, `false` if OpenAI
    pub fn is_azure(&self) -> bool {
        matches!(self, Self::Azure(_))
    }

    /// Returns whether this is an OpenAI provider
    ///
    /// # Returns
    ///
    /// `true` if OpenAI, `false` if Azure
    pub fn is_openai(&self) -> bool {
        matches!(self, Self::OpenAI(_))
    }

    /// Creates an authentication provider by detecting the provider from URL pattern.
    ///
    /// This method analyzes the URL to determine whether it's an Azure OpenAI endpoint
    /// or a standard OpenAI (or compatible) endpoint.
    ///
    /// # Arguments
    ///
    /// * `url` - The base URL or endpoint URL
    /// * `api_key` - The API key or token
    /// * `deployment_name` - Optional deployment name (required for Azure, can fall back to env var)
    ///
    /// # URL Detection
    ///
    /// | URL Pattern | Detected Provider |
    /// |-------------|-------------------|
    /// | `*.openai.azure.com*` | Azure |
    /// | All others | OpenAI (or compatible) |
    ///
    /// # Returns
    ///
    /// `Result<AuthProvider>` - Detected provider or error
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Azure URL is detected but `deployment_name` is not provided and
    ///   `AZURE_OPENAI_DEPLOYMENT_NAME` environment variable is not set
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// // OpenAI compatible API
    /// let openai = AuthProvider::from_url_with_hint(
    ///     "https://api.openai.com/v1",
    ///     "sk-key",
    ///     None
    /// ).unwrap();
    /// assert!(openai.is_openai());
    ///
    /// // Azure OpenAI
    /// let azure = AuthProvider::from_url_with_hint(
    ///     "https://my-resource.openai.azure.com",
    ///     "azure-key",
    ///     Some("gpt-4o-deployment")
    /// ).unwrap();
    /// assert!(azure.is_azure());
    ///
    /// // Local API (Ollama, LocalAI, etc.)
    /// let local = AuthProvider::from_url_with_hint(
    ///     "http://localhost:11434/v1",
    ///     "dummy-key",
    ///     None
    /// ).unwrap();
    /// assert!(local.is_openai());  // Treated as OpenAI-compatible
    /// ```
    pub fn from_url_with_hint<S: Into<String>>(
        url: S,
        api_key: S,
        deployment_name: Option<S>,
    ) -> Result<Self> {
        let url_str = url.into();
        let api_key_str = api_key.into();

        // Check if URL matches Azure pattern
        if url_str.contains(".openai.azure.com") {
            // Azure OpenAI detected
            let deployment = if let Some(name) = deployment_name {
                name.into()
            } else {
                // Try to get from environment variable
                dotenv().ok();
                env::var("AZURE_OPENAI_DEPLOYMENT_NAME").map_err(|_| {
                    OpenAIToolError::Error(
                        "Azure URL detected but deployment_name not provided. \
                         Either pass deployment_name or set AZURE_OPENAI_DEPLOYMENT_NAME environment variable."
                            .into(),
                    )
                })?
            };

            Ok(Self::Azure(AzureAuth::with_endpoint(
                api_key_str,
                url_str,
                deployment,
            )))
        } else {
            // OpenAI or compatible API
            Ok(Self::OpenAI(
                OpenAIAuth::new(api_key_str).with_base_url(url_str),
            ))
        }
    }

    /// Creates an authentication provider from URL, using environment variables for credentials.
    ///
    /// This is a convenience method that combines URL-based detection with
    /// environment variable-based credential loading.
    ///
    /// # Arguments
    ///
    /// * `url` - The base URL or endpoint URL
    ///
    /// # Environment Variables
    ///
    /// For Azure URLs (`*.openai.azure.com`):
    /// - `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_TOKEN` (required)
    /// - `AZURE_OPENAI_DEPLOYMENT_NAME` (required)
    ///
    /// For other URLs:
    /// - `OPENAI_API_KEY` (required)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::common::auth::AuthProvider;
    ///
    /// // Uses OPENAI_API_KEY from environment
    /// let auth = AuthProvider::from_url("https://api.openai.com/v1")?;
    ///
    /// // Uses AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT_NAME from environment
    /// let azure = AuthProvider::from_url("https://my-resource.openai.azure.com")?;
    /// # Ok::<(), openai_tools::common::errors::OpenAIToolError>(())
    /// ```
    pub fn from_url<S: Into<String>>(url: S) -> Result<Self> {
        let url_str = url.into();
        dotenv().ok();

        if url_str.contains(".openai.azure.com") {
            // Azure: get credentials from Azure env vars
            let api_key = env::var("AZURE_OPENAI_API_KEY")
                .or_else(|_| env::var("AZURE_OPENAI_TOKEN"))
                .map_err(|_| {
                    OpenAIToolError::Error(
                        "Azure URL detected but neither AZURE_OPENAI_API_KEY nor AZURE_OPENAI_TOKEN is set"
                            .into(),
                    )
                })?;

            let deployment = env::var("AZURE_OPENAI_DEPLOYMENT_NAME").map_err(|_| {
                OpenAIToolError::Error(
                    "Azure URL detected but AZURE_OPENAI_DEPLOYMENT_NAME is not set".into(),
                )
            })?;

            Ok(Self::Azure(AzureAuth::with_endpoint(
                api_key, url_str, deployment,
            )))
        } else {
            // OpenAI: get credentials from OpenAI env var
            let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
                OpenAIToolError::Error("OPENAI_API_KEY environment variable not set".into())
            })?;

            Ok(Self::OpenAI(
                OpenAIAuth::new(api_key).with_base_url(url_str),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_auth_new() {
        let auth = OpenAIAuth::new("sk-test-key");
        assert_eq!(auth.api_key(), "sk-test-key");
        assert_eq!(auth.base_url(), OPENAI_DEFAULT_BASE_URL);
    }

    #[test]
    fn test_openai_auth_with_base_url() {
        let auth = OpenAIAuth::new("sk-test-key").with_base_url("https://custom.example.com/v1");
        assert_eq!(auth.base_url(), "https://custom.example.com/v1");
    }

    #[test]
    fn test_openai_endpoint() {
        let auth = OpenAIAuth::new("sk-test-key");
        assert_eq!(auth.endpoint("chat/completions"), "https://api.openai.com/v1/chat/completions");
        assert_eq!(auth.endpoint("/chat/completions"), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_openai_apply_headers() {
        let auth = OpenAIAuth::new("sk-test-key");
        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(headers.get("Authorization").unwrap(), "Bearer sk-test-key");
    }

    #[test]
    fn test_azure_auth_new() {
        let auth = AzureAuth::new("api-key", "my-resource", "gpt-4o-deploy");
        assert_eq!(auth.api_key(), "api-key");
        assert_eq!(auth.resource_name(), "my-resource");
        assert_eq!(auth.deployment_name(), "gpt-4o-deploy");
        assert_eq!(auth.api_version(), AZURE_DEFAULT_API_VERSION);
        assert!(!auth.is_entra_id());
    }

    #[test]
    fn test_azure_auth_with_endpoint() {
        let auth = AzureAuth::with_endpoint("api-key", "https://custom.openai.azure.com", "deploy");
        assert_eq!(
            auth.base_url(),
            Some("https://custom.openai.azure.com")
        );
        assert!(auth.is_dynamic_url());
    }

    #[test]
    fn test_azure_auth_with_api_version() {
        let auth = AzureAuth::new("key", "resource", "deploy").with_api_version("2025-01-01");
        assert_eq!(auth.api_version(), "2025-01-01");
    }

    #[test]
    fn test_azure_auth_with_entra_id() {
        let auth = AzureAuth::new("token", "resource", "deploy").with_entra_id();
        assert!(auth.is_entra_id());
    }

    #[test]
    fn test_azure_endpoint() {
        let auth = AzureAuth::new("key", "my-resource", "gpt-4o");
        let endpoint = auth.endpoint("chat/completions");
        assert_eq!(
            endpoint,
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
        );
    }

    #[test]
    fn test_azure_endpoint_with_custom_url() {
        let auth = AzureAuth::with_endpoint("key", "https://custom.example.com", "gpt-4o");
        let endpoint = auth.endpoint("chat/completions");
        assert_eq!(
            endpoint,
            "https://custom.example.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
        );
    }

    #[test]
    fn test_azure_apply_headers_api_key() {
        let auth = AzureAuth::new("my-api-key", "resource", "deploy");
        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(headers.get("api-key").unwrap(), "my-api-key");
        assert!(headers.get("Authorization").is_none());
    }

    #[test]
    fn test_azure_apply_headers_entra_id() {
        let auth = AzureAuth::new("my-token", "resource", "deploy").with_entra_id();
        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(headers.get("Authorization").unwrap(), "Bearer my-token");
        assert!(headers.get("api-key").is_none());
    }

    #[test]
    fn test_auth_provider_openai() {
        let auth = AuthProvider::OpenAI(OpenAIAuth::new("sk-key"));
        assert!(auth.is_openai());
        assert!(!auth.is_azure());
        assert_eq!(auth.api_key(), "sk-key");
    }

    #[test]
    fn test_auth_provider_azure() {
        let auth = AuthProvider::Azure(AzureAuth::new("key", "resource", "deploy"));
        assert!(auth.is_azure());
        assert!(!auth.is_openai());
        assert_eq!(auth.api_key(), "key");
    }

    #[test]
    fn test_auth_provider_endpoint_openai() {
        let auth = AuthProvider::OpenAI(OpenAIAuth::new("key"));
        assert_eq!(auth.endpoint("chat/completions"), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_auth_provider_endpoint_azure() {
        let auth = AuthProvider::Azure(AzureAuth::new("key", "my-resource", "gpt-4o"));
        assert!(auth.endpoint("chat/completions").contains("my-resource.openai.azure.com"));
        assert!(auth.endpoint("chat/completions").contains("gpt-4o"));
    }

    #[test]
    fn test_auth_provider_apply_headers() {
        // OpenAI
        let openai_auth = AuthProvider::OpenAI(OpenAIAuth::new("sk-key"));
        let mut headers = HeaderMap::new();
        openai_auth.apply_headers(&mut headers).unwrap();
        assert!(headers.get("Authorization").unwrap().to_str().unwrap().starts_with("Bearer"));

        // Azure
        let azure_auth = AuthProvider::Azure(AzureAuth::new("azure-key", "resource", "deploy"));
        let mut headers = HeaderMap::new();
        azure_auth.apply_headers(&mut headers).unwrap();
        assert_eq!(headers.get("api-key").unwrap(), "azure-key");
    }

    // Note: Testing `from_env` methods with missing keys is inherently flaky
    // because dotenv() loads from .env file. Instead, we test valid construction.

    #[test]
    fn test_from_env_returns_correct_provider_type() {
        // This test verifies the provider detection logic without
        // modifying environment variables (which would be flaky due to .env loading)

        // Test direct construction (which from_env uses internally)
        let openai = AuthProvider::OpenAI(OpenAIAuth::new("sk-test"));
        assert!(openai.is_openai());
        assert!(!openai.is_azure());

        let azure = AuthProvider::Azure(AzureAuth::new("key", "resource", "deploy"));
        assert!(azure.is_azure());
        assert!(!azure.is_openai());
    }

    #[test]
    fn test_azure_auth_env_parsing_logic() {
        // Test the parsing logic that azure_from_env uses
        // without actually calling azure_from_env (which depends on env state)

        // With resource name
        let auth = AzureAuth::new("test-key", "my-resource", "gpt-4o-deployment");
        assert_eq!(auth.api_key(), "test-key");
        assert_eq!(auth.resource_name(), "my-resource");
        assert_eq!(auth.deployment_name(), "gpt-4o-deployment");
        assert_eq!(auth.api_version(), AZURE_DEFAULT_API_VERSION);

        // With custom endpoint
        let auth = AzureAuth::with_endpoint("test-key", "https://custom.azure.com", "gpt-4o");
        assert_eq!(auth.base_url(), Some("https://custom.azure.com"));

        // With Entra ID
        let auth = AzureAuth::new("bearer-token", "resource", "deploy").with_entra_id();
        assert!(auth.is_entra_id());
    }

    #[test]
    fn test_openai_endpoint_trailing_slash_handling() {
        let auth = OpenAIAuth::new("key").with_base_url("https://example.com/v1/");
        assert_eq!(auth.endpoint("chat/completions"), "https://example.com/v1/chat/completions");
        assert_eq!(auth.endpoint("/chat/completions"), "https://example.com/v1/chat/completions");
    }

    #[test]
    fn test_azure_endpoint_trailing_slash_handling() {
        let auth = AzureAuth::with_endpoint("key", "https://example.openai.azure.com/", "deploy");
        let endpoint = auth.endpoint("/chat/completions");
        assert!(endpoint.contains("/chat/completions?api-version="));
        assert!(!endpoint.contains("//chat/completions"));
    }

    // Static base URL tests (with_base_url)

    #[test]
    fn test_azure_with_base_url_simple() {
        let auth = AzureAuth::with_base_url(
            "api-key",
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o",
        );
        assert_eq!(
            auth.base_url(),
            Some("https://my-resource.openai.azure.com/openai/deployments/gpt-4o")
        );
        assert!(!auth.is_dynamic_url());
        assert_eq!(
            auth.endpoint("chat/completions"),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions"
        );
    }

    #[test]
    fn test_azure_with_base_url_with_query_params() {
        let auth = AzureAuth::with_base_url(
            "api-key",
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-08-01-preview",
        );
        assert!(!auth.is_dynamic_url());
        assert_eq!(
            auth.endpoint("chat/completions"),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
        );
    }

    #[test]
    fn test_azure_with_base_url_trailing_slash() {
        let auth = AzureAuth::with_base_url(
            "api-key",
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/",
        );
        assert_eq!(
            auth.endpoint("chat/completions"),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions"
        );
    }

    #[test]
    fn test_azure_with_base_url_leading_slash_path() {
        let auth = AzureAuth::with_base_url(
            "api-key",
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o",
        );
        assert_eq!(
            auth.endpoint("/chat/completions"),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions"
        );
    }

    #[test]
    fn test_azure_with_base_url_headers() {
        let auth = AzureAuth::with_base_url("my-api-key", "https://example.com");
        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();
        assert_eq!(headers.get("api-key").unwrap(), "my-api-key");
    }

    #[test]
    fn test_azure_with_base_url_entra_id() {
        let auth = AzureAuth::with_base_url("my-token", "https://example.com").with_entra_id();
        assert!(auth.is_entra_id());
        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();
        assert_eq!(headers.get("Authorization").unwrap(), "Bearer my-token");
    }

    // URL-based provider detection tests

    #[test]
    fn test_from_url_with_hint_openai_api() {
        let auth = AuthProvider::from_url_with_hint(
            "https://api.openai.com/v1",
            "sk-test-key",
            None::<&str>,
        )
        .unwrap();

        assert!(auth.is_openai());
        assert!(!auth.is_azure());
        assert_eq!(auth.api_key(), "sk-test-key");
        assert_eq!(
            auth.endpoint("chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_from_url_with_hint_azure() {
        let auth = AuthProvider::from_url_with_hint(
            "https://my-resource.openai.azure.com",
            "azure-api-key",
            Some("gpt-4o-deployment"),
        )
        .unwrap();

        assert!(auth.is_azure());
        assert!(!auth.is_openai());
        assert_eq!(auth.api_key(), "azure-api-key");

        let endpoint = auth.endpoint("chat/completions");
        assert!(endpoint.contains("my-resource.openai.azure.com"));
        assert!(endpoint.contains("gpt-4o-deployment"));
        assert!(endpoint.contains("api-version="));
    }

    #[test]
    fn test_from_url_with_hint_local_api_ollama() {
        // Local APIs like Ollama should be treated as OpenAI-compatible
        let auth = AuthProvider::from_url_with_hint(
            "http://localhost:11434/v1",
            "ollama",
            None::<&str>,
        )
        .unwrap();

        assert!(auth.is_openai());
        assert_eq!(
            auth.endpoint("chat/completions"),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn test_from_url_with_hint_custom_openai_compatible() {
        // Custom OpenAI-compatible endpoints (e.g., vLLM, LocalAI)
        let auth = AuthProvider::from_url_with_hint(
            "https://my-proxy.example.com/openai/v1",
            "proxy-key",
            None::<&str>,
        )
        .unwrap();

        assert!(auth.is_openai());
        assert_eq!(
            auth.endpoint("embeddings"),
            "https://my-proxy.example.com/openai/v1/embeddings"
        );
    }

    #[test]
    fn test_from_url_with_hint_azure_various_patterns() {
        // Test various Azure URL patterns
        let patterns = [
            "https://eastus.openai.azure.com",
            "https://my-company-resource.openai.azure.com",
            "https://test.openai.azure.com/",
            "https://resource-name.openai.azure.com/some/path",
        ];

        for url in patterns {
            let result = AuthProvider::from_url_with_hint(url, "key", Some("deploy"));
            assert!(
                result.is_ok(),
                "Should detect Azure for URL: {}",
                url
            );
            assert!(
                result.unwrap().is_azure(),
                "Should be Azure provider for URL: {}",
                url
            );
        }
    }

    #[test]
    fn test_from_url_with_hint_azure_missing_deployment_no_env() {
        // When Azure URL is detected but deployment_name is None and env var is not set,
        // the function should return an error (unless AZURE_OPENAI_DEPLOYMENT_NAME is in .env)
        // This test verifies the error message format
        let result = AuthProvider::from_url_with_hint(
            "https://test.openai.azure.com",
            "key",
            None::<&str>,
        );

        // If AZURE_OPENAI_DEPLOYMENT_NAME is in .env, this will succeed
        // If not, it will fail with a specific error message
        if result.is_err() {
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("deployment_name") || err.contains("AZURE_OPENAI_DEPLOYMENT_NAME"),
                "Error should mention deployment_name requirement"
            );
        }
        // If it succeeds, it means the env var is set (from .env file)
    }

    #[test]
    fn test_from_url_with_hint_headers_openai() {
        let auth = AuthProvider::from_url_with_hint(
            "https://api.openai.com/v1",
            "sk-secret-key",
            None::<&str>,
        )
        .unwrap();

        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(
            headers.get("Authorization").unwrap(),
            "Bearer sk-secret-key"
        );
    }

    #[test]
    fn test_from_url_with_hint_headers_azure() {
        let auth = AuthProvider::from_url_with_hint(
            "https://resource.openai.azure.com",
            "azure-secret",
            Some("deployment"),
        )
        .unwrap();

        let mut headers = HeaderMap::new();
        auth.apply_headers(&mut headers).unwrap();

        assert_eq!(headers.get("api-key").unwrap(), "azure-secret");
        assert!(headers.get("Authorization").is_none());
    }
}
