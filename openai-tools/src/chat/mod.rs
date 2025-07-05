pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::chat::request::ChatCompletion;
    use crate::common::{errors::OpenAIToolError, message::Message, role::Role, structured_output::Schema};
    use serde::{Deserialize, Serialize};
    use serde_json;
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
    async fn test_chat_completion() {
        init_tracing();
        let mut chat = ChatCompletion::new();
        let messages = vec![Message::from_string(Role::User, "Hi there!")];

        chat.model_id("gpt-4o-mini").messages(messages).temperature(1.0);

        let mut counter = 3;
        loop {
            match chat.chat().await {
                Ok(response) => {
                    tracing::info!("{}", &response.choices[0].message.content);
                    assert!(true);
                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            assert!(false, "Chat completion failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        assert!(false, "Chat completion failed");
                    }
                },
            };
        }
    }

    #[tokio::test]
    async fn test_chat_completion_2() {
        init_tracing();
        let mut chat = ChatCompletion::new();
        let messages = vec![Message::from_string(Role::User, "トンネルを抜けると？")];

        chat.model_id("gpt-4o-mini").messages(messages).temperature(1.5);

        let mut counter = 3;
        loop {
            match chat.chat().await {
                Ok(response) => {
                    println!("{}", &response.choices[0].message.content);
                    assert!(true);
                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            assert!(false, "Chat completion failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        assert!(false, "Chat completion failed");
                    }
                },
            };
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Weather {
        #[serde(default = "String::new")]
        location: String,
        #[serde(default = "String::new")]
        date: String,
        #[serde(default = "String::new")]
        weather: String,
        #[serde(default = "String::new")]
        error: String,
    }

    #[tokio::test]
    async fn test_chat_completion_with_json_schema() {
        init_tracing();
        let mut openai = ChatCompletion::new();
        let messages = vec![Message::from_string(Role::User, "Hi there! How's the weather tomorrow in Tokyo? If you can't answer, report error.")];

        let mut json_schema = Schema::chat_json_schema("weather");
        json_schema.add_property("location", "string", "The location to check the weather for.");
        json_schema.add_property("date", "string", "The date to check the weather for.");
        json_schema.add_property("weather", "string", "The weather for the location and date.");
        json_schema.add_property("error", "string", "Error message. If there is no error, leave this field empty.");
        openai.model_id("gpt-4o-mini").messages(messages).temperature(1.0).json_schema(json_schema);

        let mut counter = 3;
        loop {
            match openai.chat().await {
                Ok(response) => {
                    println!("{:#?}", response);
                    match serde_json::from_str::<Weather>(&response.choices[0].message.content) {
                        Ok(weather) => {
                            println!("{:#?}", weather);
                            assert!(true);
                        }
                        Err(e) => {
                            println!("{:#?}", e);
                            assert!(false);
                        }
                    }
                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            assert!(false, "Chat completion failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        assert!(false, "Chat completion failed");
                    }
                },
            };
        }
    }

    #[derive(Deserialize)]
    struct Summary {
        pub is_survey: bool,
        pub research_question: String,
        pub contributions: String,
        pub dataset: String,
        pub proposed_method: String,
        pub experiment_results: String,
        pub comparison_with_related_works: String,
        pub future_works: String,
    }
    #[tokio::test]
    async fn test_summarize() {
        init_tracing();
        let mut openai = ChatCompletion::new();
        let instruction = std::fs::read_to_string("src/test_rsc/sample_instruction.txt").unwrap();

        let messages = vec![Message::from_string(Role::User, instruction.clone())];

        let mut json_schema = Schema::chat_json_schema("summary");
        json_schema.add_property("is_survey", "boolean", "この論文がサーベイ論文かどうかをtrue/falseで判定．");
        json_schema.add_property("research_question", "string", "この論文のリサーチクエスチョンの説明．この論文の背景や既存研究との関連も含めて記述する．");
        json_schema.add_property("contributions", "string", "この論文のコントリビューションをリスト形式で記述する．");
        json_schema.add_property("dataset", "string", "この論文で使用されているデータセットをリストアップする．");
        json_schema.add_property("proposed_method", "string", "提案手法の詳細な説明．");
        json_schema.add_property("experiment_results", "string", "実験の結果の詳細な説明．");
        json_schema.add_property(
            "comparison_with_related_works",
            "string",
            "関連研究と比較した場合のこの論文の新規性についての説明．可能な限り既存研究を参照しながら記述すること．",
        );
        json_schema.add_property("future_works", "string", "未解決の課題および将来の研究の方向性について記述．");

        openai.model_id(String::from("gpt-4o-mini")).messages(messages).temperature(1.0).json_schema(json_schema);

        let mut counter = 3;
        loop {
            match openai.chat().await {
                Ok(response) => {
                    println!("{:#?}", response);
                    match serde_json::from_str::<Summary>(&response.choices[0].message.content) {
                        Ok(summary) => {
                            tracing::info!("Summary.is_survey: {}", summary.is_survey);
                            tracing::info!("Summary.research_question: {}", summary.research_question);
                            tracing::info!("Summary.contributions: {}", summary.contributions);
                            tracing::info!("Summary.dataset: {}", summary.dataset);
                            tracing::info!("Summary.proposed_method: {}", summary.proposed_method);
                            tracing::info!("Summary.experiment_results: {}", summary.experiment_results);
                            tracing::info!("Summary.comparison_with_related_works: {}", summary.comparison_with_related_works);
                            tracing::info!("Summary.future_works: {}", summary.future_works);
                            assert!(true);
                        }
                        Err(e) => {
                            tracing::error!("Error: {}", e);
                            assert!(false);
                        }
                    }
                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            assert!(false, "Chat completion failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        assert!(false, "Chat completion failed");
                    }
                },
            };
        }
    }

    // #[tokio::test]
    // async fn test_chat_completion_with_long_arguments() {
    //     init_tracing();
    //     let mut openai = ChatCompletion::new();
    //     let text = std::fs::read_to_string("src/test_rsc/long_text.txt").unwrap();
    //     let messages = vec![Message::from_string(Role::User, text)];

    //     let token_count = messages
    //         .iter()
    //         .map(|m| m.get_input_token_count())
    //         .sum::<usize>();
    //     tracing::info!("Token count: {}", token_count);

    //     openai
    //         .model_id(String::from("gpt-4o-mini"))
    //         .messages(messages)
    //         .temperature(1.0);

    //     let mut counter = 3;
    //     loop {
    //         match openai.chat().await {
    //             Ok(response) => {
    //                 println!("{:#?}", response);
    //                 assert!(true);
    //                 break;
    //             }
    //             Err(e) => match e {
    //                 OpenAIToolError::RequestError(e) => {
    //                     tracing::warn!("Request error: {} (retrying... {})", e, counter);
    //                     counter -= 1;
    //                     if counter == 0 {
    //                         assert!(false, "Chat completion failed (retry limit reached)");
    //                     }
    //                     continue;
    //                 }
    //                 _ => {
    //                     tracing::error!("Error: {}", e);
    //                     assert!(false, "Chat completion failed");
    //                 }
    //             },
    //         };
    //     }
    // }
}
