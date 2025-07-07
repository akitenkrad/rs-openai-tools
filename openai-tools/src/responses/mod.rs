pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::common::{
        message::{Content, Message},
        role::Role,
        structured_output::Schema,
        tool::{ParameterProp, Tool},
    };
    use crate::responses::request::Responses;

    use serde::Deserialize;
    use std::sync::Once;
    use tracing_subscriber::EnvFilter;

    static INIT: Once = Once::new();

    fn init_tracing() {
        INIT.call_once(|| {
            // `RUST_LOG` 環境変数があればそれを使い、なければ "info"
            let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer() // `cargo test` / nextest 用
                .init();
        });
    }

    #[tokio::test]
    async fn test_responses_with_plain_text() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini");
        responses.instructions("test instructions");
        responses.plain_text_input("Hello world!");

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        tracing::info!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_responses_with_messages() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini");
        responses.instructions("test instructions");
        let messages = vec![Message::from_string(Role::User, "Hello world!")];
        responses.messages(messages);

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        tracing::info!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_responses_with_tools() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini");
        responses.instructions("test instructions");
        let messages = vec![Message::from_string(Role::User, "Calculate 2 + 2 using a calculator tool.")];
        responses.messages(messages);

        let tool = Tool::function(
            "calculator",
            "A simple calculator tool",
            vec![("a", ParameterProp::number("The first number")), ("b", ParameterProp::number("The second number"))],
            false,
        );
        responses.tools(vec![tool]);

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        println!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert_eq!(res.output[0].type_name, "function_call");
                    assert_eq!(res.output[0].name.as_ref().unwrap(), "calculator");
                    assert!(res.output[0].call_id.as_ref().unwrap().len() > 0);
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }

    #[derive(Debug, Deserialize)]
    struct TestResponse {
        pub capital: String,
    }
    #[tokio::test]
    async fn test_responses_with_json_schema() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini");

        let messages = vec![Message::from_string(Role::User, "What is the capital of France?")];
        responses.messages(messages);

        let mut schema = Schema::responses_json_schema("capital");
        schema.add_property("capital", "string", "The capital city of France");
        responses.text(schema);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    let res = serde_json::from_str::<TestResponse>(res.output[0].content.as_ref().unwrap()[0].text.as_str()).unwrap();
                    assert_eq!(res.capital, "Paris");
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_responses_with_image_input() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini");
        responses.instructions("test instructions");

        let message = Message::from_message_array(
            Role::User,
            vec![Content::from_text("Do you find a clock in this image?"), Content::from_image_file("src/test_rsc/sample_image.jpg")],
        );
        responses.messages(vec![message]);

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        tracing::info!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }
}
