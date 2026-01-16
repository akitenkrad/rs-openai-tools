//! OpenAI Fine-tuning API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Fine-tuning API.
//! It allows you to create, list, retrieve, and cancel fine-tuning jobs, as well as
//! access training events and checkpoints.
//!
//! # Key Features
//!
//! - **Create Jobs**: Start a fine-tuning job with custom hyperparameters
//! - **Retrieve Jobs**: Get the status and details of a fine-tuning job
//! - **List Jobs**: List all fine-tuning jobs
//! - **Cancel Jobs**: Cancel an in-progress job
//! - **List Events**: View training progress and events
//! - **List Checkpoints**: Access model checkpoints from training
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::fine_tuning::request::{FineTuning, CreateFineTuningJobRequest};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let fine_tuning = FineTuning::new()?;
//!
//!     // List all fine-tuning jobs
//!     let response = fine_tuning.list(None, None).await?;
//!     for job in &response.data {
//!         println!("{}: {:?}", job.id, job.status);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::common::auth::AuthProvider;
use crate::common::client::create_http_client;
use crate::common::errors::{OpenAIToolError, Result};
use crate::common::models::FineTuningModel;
use crate::fine_tuning::response::{
    DpoConfig, FineTuningCheckpointListResponse, FineTuningEventListResponse, FineTuningJob,
    FineTuningJobListResponse, Hyperparameters, Integration, MethodConfig, SupervisedConfig,
};
use serde::Serialize;
use std::time::Duration;

/// Default API path for Fine-tuning
const FINE_TUNING_PATH: &str = "fine_tuning/jobs";

/// Request to create a new fine-tuning job.
#[derive(Debug, Clone, Serialize)]
pub struct CreateFineTuningJobRequest {
    /// The base model to fine-tune.
    pub model: FineTuningModel,

    /// The ID of the uploaded training file.
    pub training_file: String,

    /// The ID of the uploaded validation file (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,

    /// A string suffix for the fine-tuned model name (max 64 chars).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    /// A seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// The fine-tuning method and hyperparameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<MethodConfig>,

    /// Integrations to enable (e.g., Weights & Biases).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integrations: Option<Vec<Integration>>,
}

impl CreateFineTuningJobRequest {
    /// Creates a new fine-tuning job request with the given model and training file.
    ///
    /// # Arguments
    ///
    /// * `model` - The base model to fine-tune
    /// * `training_file` - The ID of the uploaded training file
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::fine_tuning::request::CreateFineTuningJobRequest;
    /// use openai_tools::common::models::FineTuningModel;
    ///
    /// let request = CreateFineTuningJobRequest::new(
    ///     FineTuningModel::Gpt4oMini_2024_07_18,
    ///     "file-abc123"
    /// );
    /// ```
    pub fn new(model: FineTuningModel, training_file: impl Into<String>) -> Self {
        Self {
            model,
            training_file: training_file.into(),
            validation_file: None,
            suffix: None,
            seed: None,
            method: None,
            integrations: None,
        }
    }

    /// Sets the validation file for the job.
    pub fn with_validation_file(mut self, file_id: impl Into<String>) -> Self {
        self.validation_file = Some(file_id.into());
        self
    }

    /// Sets the suffix for the fine-tuned model name.
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    /// Sets the seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Configures supervised fine-tuning with custom hyperparameters.
    pub fn with_supervised_method(mut self, hyperparameters: Option<Hyperparameters>) -> Self {
        self.method = Some(MethodConfig {
            method_type: "supervised".to_string(),
            supervised: Some(SupervisedConfig { hyperparameters }),
            dpo: None,
        });
        self
    }

    /// Configures DPO (Direct Preference Optimization) fine-tuning.
    pub fn with_dpo_method(mut self, hyperparameters: Option<Hyperparameters>) -> Self {
        self.method = Some(MethodConfig {
            method_type: "dpo".to_string(),
            supervised: None,
            dpo: Some(DpoConfig { hyperparameters }),
        });
        self
    }

    /// Adds integrations to the job.
    pub fn with_integrations(mut self, integrations: Vec<Integration>) -> Self {
        self.integrations = Some(integrations);
        self
    }
}

/// Client for interacting with the OpenAI Fine-tuning API.
///
/// This struct provides methods to create, list, retrieve, and cancel fine-tuning jobs,
/// as well as access training events and checkpoints.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::fine_tuning::request::{FineTuning, CreateFineTuningJobRequest};
/// use openai_tools::fine_tuning::response::Hyperparameters;
/// use openai_tools::common::models::FineTuningModel;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let fine_tuning = FineTuning::new()?;
///
///     // Create a fine-tuning job
///     let hyperparams = Hyperparameters {
///         n_epochs: Some(3),
///         batch_size: None,
///         learning_rate_multiplier: None,
///     };
///
///     let request = CreateFineTuningJobRequest::new(
///             FineTuningModel::Gpt4oMini_2024_07_18,
///             "file-abc123"
///         )
///         .with_suffix("my-custom-model")
///         .with_supervised_method(Some(hyperparams));
///
///     let job = fine_tuning.create(request).await?;
///     println!("Created job: {} ({:?})", job.id, job.status);
///
///     Ok(())
/// }
/// ```
pub struct FineTuning {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl FineTuning {
    /// Creates a new FineTuning client for OpenAI API.
    ///
    /// Initializes the client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(FineTuning)` - A new FineTuning client ready for use
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::fine_tuning::request::FineTuning;
    ///
    /// let fine_tuning = FineTuning::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        let auth = AuthProvider::openai_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new FineTuning client with a custom authentication provider
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, timeout: None }
    }

    /// Creates a new FineTuning client for Azure OpenAI API
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new FineTuning client by auto-detecting the provider
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new FineTuning client with URL-based provider detection
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, timeout: None }
    }

    /// Creates a new FineTuning client from URL using environment variables
    pub fn from_url<S: Into<String>>(url: S) -> Result<Self> {
        let auth = AuthProvider::from_url(url)?;
        Ok(Self { auth, timeout: None })
    }

    /// Returns the authentication provider
    pub fn auth(&self) -> &AuthProvider {
        &self.auth
    }

    /// Sets the request timeout duration.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The maximum time to wait for a response
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Creates the HTTP client with default headers.
    fn create_client(&self) -> Result<(request::Client, request::header::HeaderMap)> {
        let client = create_http_client(self.timeout)?;
        let mut headers = request::header::HeaderMap::new();
        self.auth.apply_headers(&mut headers)?;
        headers.insert(
            "Content-Type",
            request::header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            "User-Agent",
            request::header::HeaderValue::from_static("openai-tools-rust"),
        );
        Ok((client, headers))
    }

    /// Creates a new fine-tuning job.
    ///
    /// # Arguments
    ///
    /// * `request` - The fine-tuning job creation request
    ///
    /// # Returns
    ///
    /// * `Ok(FineTuningJob)` - The created job object
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::fine_tuning::request::{FineTuning, CreateFineTuningJobRequest};
    /// use openai_tools::common::models::FineTuningModel;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let fine_tuning = FineTuning::new()?;
    ///
    ///     let request = CreateFineTuningJobRequest::new(
    ///             FineTuningModel::Gpt4oMini_2024_07_18,
    ///             "file-abc123"
    ///         )
    ///         .with_suffix("my-model");
    ///
    ///     let job = fine_tuning.create(request).await?;
    ///     println!("Created job: {}", job.id);
    ///     Ok(())
    /// }
    /// ```
    pub async fn create(&self, request: CreateFineTuningJobRequest) -> Result<FineTuningJob> {
        let (client, headers) = self.create_client()?;

        let body = serde_json::to_string(&request).map_err(OpenAIToolError::SerdeJsonError)?;

        let url = self.auth.endpoint(FINE_TUNING_PATH);
        let response = client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<FineTuningJob>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Retrieves details of a specific fine-tuning job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The ID of the job to retrieve
    ///
    /// # Returns
    ///
    /// * `Ok(FineTuningJob)` - The job details
    /// * `Err(OpenAIToolError)` - If the job is not found or the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::fine_tuning::request::FineTuning;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let fine_tuning = FineTuning::new()?;
    ///     let job = fine_tuning.retrieve("ftjob-abc123").await?;
    ///
    ///     println!("Status: {:?}", job.status);
    ///     if let Some(model) = &job.fine_tuned_model {
    ///         println!("Fine-tuned model: {}", model);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn retrieve(&self, job_id: &str) -> Result<FineTuningJob> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", self.auth.endpoint(FINE_TUNING_PATH), job_id);

        let response = client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<FineTuningJob>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Cancels an in-progress fine-tuning job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The ID of the job to cancel
    ///
    /// # Returns
    ///
    /// * `Ok(FineTuningJob)` - The updated job object
    /// * `Err(OpenAIToolError)` - If the job cannot be cancelled or the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::fine_tuning::request::FineTuning;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let fine_tuning = FineTuning::new()?;
    ///     let job = fine_tuning.cancel("ftjob-abc123").await?;
    ///
    ///     println!("Job status: {:?}", job.status);
    ///     Ok(())
    /// }
    /// ```
    pub async fn cancel(&self, job_id: &str) -> Result<FineTuningJob> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}/cancel", self.auth.endpoint(FINE_TUNING_PATH), job_id);

        let response = client
            .post(&url)
            .headers(headers)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<FineTuningJob>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Lists all fine-tuning jobs.
    ///
    /// Supports pagination through `limit` and `after` parameters.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of jobs to return (default: 20)
    /// * `after` - Cursor for pagination (job ID to start after)
    ///
    /// # Returns
    ///
    /// * `Ok(FineTuningJobListResponse)` - The list of jobs
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::fine_tuning::request::FineTuning;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let fine_tuning = FineTuning::new()?;
    ///
    ///     let response = fine_tuning.list(Some(10), None).await?;
    ///     for job in &response.data {
    ///         println!("{}: {:?}", job.id, job.status);
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn list(
        &self,
        limit: Option<u32>,
        after: Option<&str>,
    ) -> Result<FineTuningJobListResponse> {
        let (client, headers) = self.create_client()?;

        let mut url = self.auth.endpoint(FINE_TUNING_PATH);
        let mut params = Vec::new();

        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(a) = after {
            params.push(format!("after={}", a));
        }

        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }

        let response = client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<FineTuningJobListResponse>(&content)
            .map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Lists events for a fine-tuning job.
    ///
    /// Events provide insight into the training process.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The ID of the fine-tuning job
    /// * `limit` - Maximum number of events to return (default: 20)
    /// * `after` - Cursor for pagination (event ID to start after)
    ///
    /// # Returns
    ///
    /// * `Ok(FineTuningEventListResponse)` - The list of events
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::fine_tuning::request::FineTuning;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let fine_tuning = FineTuning::new()?;
    ///
    ///     let response = fine_tuning.list_events("ftjob-abc123", Some(10), None).await?;
    ///     for event in &response.data {
    ///         println!("[{}] {}: {}", event.level, event.event_type, event.message);
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn list_events(
        &self,
        job_id: &str,
        limit: Option<u32>,
        after: Option<&str>,
    ) -> Result<FineTuningEventListResponse> {
        let (client, headers) = self.create_client()?;

        let mut url = format!("{}/{}/events", self.auth.endpoint(FINE_TUNING_PATH), job_id);
        let mut params = Vec::new();

        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(a) = after {
            params.push(format!("after={}", a));
        }

        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }

        let response = client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<FineTuningEventListResponse>(&content)
            .map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Lists checkpoints for a fine-tuning job.
    ///
    /// Checkpoints are saved at the end of each training epoch.
    /// Only the last 3 checkpoints are available.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The ID of the fine-tuning job
    /// * `limit` - Maximum number of checkpoints to return (default: 10)
    /// * `after` - Cursor for pagination (checkpoint ID to start after)
    ///
    /// # Returns
    ///
    /// * `Ok(FineTuningCheckpointListResponse)` - The list of checkpoints
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::fine_tuning::request::FineTuning;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let fine_tuning = FineTuning::new()?;
    ///
    ///     let response = fine_tuning.list_checkpoints("ftjob-abc123", None, None).await?;
    ///     for checkpoint in &response.data {
    ///         println!("Step {}: loss={}", checkpoint.step_number, checkpoint.metrics.train_loss);
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn list_checkpoints(
        &self,
        job_id: &str,
        limit: Option<u32>,
        after: Option<&str>,
    ) -> Result<FineTuningCheckpointListResponse> {
        let (client, headers) = self.create_client()?;

        let mut url = format!("{}/{}/checkpoints", self.auth.endpoint(FINE_TUNING_PATH), job_id);
        let mut params = Vec::new();

        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(a) = after {
            params.push(format!("after={}", a));
        }

        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }

        let response = client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<FineTuningCheckpointListResponse>(&content)
            .map_err(OpenAIToolError::SerdeJsonError)
    }
}
