use super::json_schema::*;
use super::*;

#[test]
fn test_build_json_schema_simple() {
    let mut json_schema = JsonSchema::new("test-schema");
    json_schema.add_property("test_property", "string", None);

    let schema_string = serde_json::to_string(&json_schema).unwrap();
    println!("{}", serde_json::to_string_pretty(&json_schema).unwrap());

    assert_eq!(
        schema_string,
        r#"{"name":"test-schema","schema":{"type":"object","properties":{"test_property":{"type":"string"}},"required":["test_property"],"additionalProperties":false}}"#
    );
}

#[test]
fn test_build_json_schema_with_description() {
    let mut json_schema = JsonSchema::new("test-schema");
    json_schema.add_property(
        "email",
        "string",
        Some(String::from("The email address that appears in the input")),
    );

    let schema_string = serde_json::to_string(&json_schema).unwrap();
    println!("{}", serde_json::to_string_pretty(&json_schema).unwrap());

    assert_eq!(
        schema_string,
        r#"{"name":"test-schema","schema":{"type":"object","properties":{"email":{"type":"string","description":"The email address that appears in the input"}},"required":["email"],"additionalProperties":false}}"#,
    );
}

#[test]
fn test_build_json_schema_add_array() {
    let mut json_schema = JsonSchema::new("test-schema");
    json_schema.add_property(
        "test-property",
        "string",
        Some(String::from("This is a test property")),
    );
    json_schema.add_array(
        "test-array",
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

#[test]
fn test_build_body_simple() {
    let body = ChatCompletionRequestBody::new(
        "gpt-4o-mini".to_string(),
        vec![Message::new(
            "assistant",
            "Hi there! How can I assist you today?",
        )],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let body_string = serde_json::to_string(&body).unwrap();
    println!("{}", serde_json::to_string_pretty(&body).unwrap());

    assert_eq!(
        body_string,
        r#"{"model":"gpt-4o-mini","messages":[{"role":"assistant","content":"Hi there! How can I assist you today?"}]}"#
    );
}

#[test]
fn test_build_body_with_args() {
    let mut json_schema = JsonSchema::new("test-schema");
    json_schema.add_property(
        "test_property",
        "string",
        Option::from(String::from("This is a test property")),
    );
    let response_format = ResponseFormat::new("json_schema", json_schema);
    let body = ChatCompletionRequestBody::new(
        "gpt-4o-mini".to_string(),
        vec![Message::new(
            "assistant",
            "Hi there! How can I assist you today?",
        )],
        Some(true),
        Some(0.5),
        Some(FxHashMap::default()),
        Some(true),
        Some(10),
        Some(1000),
        Some(10),
        Some(vec!["test".to_string()]),
        Some(0.5),
        Some(0.5),
        Some(response_format),
    );

    let body_string = serde_json::to_string(&body).unwrap();
    println!("{}", body_string);

    assert_eq!(
        body_string,
        r#"{"model":"gpt-4o-mini","messages":[{"role":"assistant","content":"Hi there! How can I assist you today?"}],"store":true,"frequency_penalty":0.5,"logit_bias":{},"logprobs":true,"top_logprobs":10,"max_completion_tokens":1000,"n":10,"modalities":["test"],"presence_penalty":0.5,"temperature":0.5,"response_format":{"type":"json_schema","json_schema":{"name":"test-schema","schema":{"type":"object","properties":{"test_property":{"type":"string","description":"This is a test property"}},"required":["test_property"],"additionalProperties":false}}}}"#
    );
}

#[test]
fn test_deserialize_api_response() {
    let response = r#"{
    "id": "chatcmpl-123456",
    "object": "chat.completion",
    "created": 1728933352,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hi there! How can I assist you today?",
                "refusal": "No",
                "tool_calls": [
                    {
                        "id": "tool-123",
                        "type": "function",
                        "function": {
                            "name": "get_user_info",
                            "arguments": "user_id"
                        }
                    }
                ]
            },
            "logprobs": {
                "content": [
                    {
                        "token": "Hi",
                        "logprob": -0.0001,
                        "bytes": [0, 1, 2, 3],
                        "top_logprobs": [
                            {
                                "token": "Hi",
                                "logprob": -0.0001,
                                "bytes": [0, 1, 2, 3]
                            }
                        ]
                    }
                ],
                "refusal": [
                    {
                        "token": "No",
                        "logprob": -0.0001,
                        "bytes": [0, 1, 2, 3],
                        "top_logprobs": [
                            {
                                "token": "No",
                                "logprob": -0.0001,
                                "bytes": [0, 1, 2, 3]
                            }
                        ]
                    }
                ]
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 19,
        "completion_tokens": 10,
        "total_tokens": 29,
        "prompt_tokens_details": {
            "cached_tokens": 1
        },
        "completion_tokens_details": {
            "reasoning_tokens": 2,
            "accepted_prediction_tokens": 3,
            "rejected_prediction_tokens": 4
        }
    },
    "system_fingerprint": "fp_6b68a8204b"
}"#;

    let response: Response = serde_json::from_str(response).unwrap();
    println!("{:#?}", response);
    assert!(true);
}

#[test]
fn test_chat_completion() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new("user", "Hi there!")];

    openai
        .model_id("gpt-4o-mini")
        .messages(messages)
        .temperature(1.0);

    let response = openai.chat().unwrap();
    println!("{}", &response.choices[0].message.content);
    assert!(true);
}

#[test]
fn test_chat_completion_2() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new("user", "トンネルを抜けると？")];

    openai
        .model_id("gpt-4o-mini")
        .messages(messages)
        .temperature(1.5);

    let response = openai.chat().unwrap();
    println!("{}", &response.choices[0].message.content);
    assert!(true);
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

#[test]
fn test_chat_completion_with_json_schema() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new(
        "user",
        "Hi there! How's the weather tomorrow in Tokyo? If you can't answer, report error.",
    )];

    let mut json_schema = JsonSchema::new("weather");
    json_schema.add_property(
        "location",
        "string",
        Option::from(String::from("The location to check the weather for.")),
    );
    json_schema.add_property(
        "date",
        "string",
        Option::from(String::from("The date to check the weather for.")),
    );
    json_schema.add_property(
        "weather",
        "string",
        Option::from(String::from("The weather for the location and date.")),
    );
    json_schema.add_property(
        "error",
        "string",
        Option::from(String::from(
            "Error message. If there is no error, leave this field empty.",
        )),
    );
    openai
        .model_id("gpt-4o-mini")
        .messages(messages)
        .temperature(1.0)
        .response_format(ResponseFormat::new("json_schema", json_schema));

    let response = openai.chat().unwrap();
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
}

#[test]
fn test_chat_completion_with_json_schema_expect_error() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new(
        "user",
        "Hi there! How's the weather tomorrow in Tokyo? Today is 3024/12/25. If you can't answer, report error."
            ,
    )];

    let mut json_schema = JsonSchema::new("weather");
    json_schema.add_property(
        "location",
        "string",
        Option::from(String::from("The location to check the weather for.")),
    );
    json_schema.add_property(
        "date",
        "string",
        Option::from(String::from("The date to check the weather for.")),
    );
    json_schema.add_property(
        "weather",
        "string",
        Option::from(String::from("The weather for the location and date.")),
    );
    json_schema.add_property(
        "error",
        "string",
        Option::from(String::from(
            "Error message. If there is no error, leave this field empty.",
        )),
    );
    openai
        .model_id("gpt-4o-mini")
        .messages(messages)
        .temperature(1.0)
        .response_format(ResponseFormat::new("json_schema", json_schema));

    let response = openai.chat().unwrap();
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
}

#[test]
fn test_summarize() {
    let mut openai = OpenAI::new();
    let instruction = std::fs::read_to_string("src/test_rsc/sample_instruction.txt").unwrap();

    let messages = vec![Message::new("user", &instruction)];

    let mut json_schema = JsonSchema::new("summary");
    json_schema.add_property(
        "is_survey",
        "boolean",
        Option::from(String::from(
            "この論文がサーベイ論文かどうかをtrue/falseで判定．",
        )),
    );
    json_schema.add_property(
        "research_question",
        "string",
        Option::from(String::from("この論文のリサーチクエスチョンの説明．この論文の背景や既存研究との関連も含めて記述する．")),
    );
    json_schema.add_property(
        "contributions",
        "string",
        Option::from(String::from(
            "この論文のコントリビューションをリスト形式で記述する．",
        )),
    );
    json_schema.add_property(
        "dataset",
        "string",
        Option::from(String::from(
            "この論文で使用されているデータセットをリストアップする．",
        )),
    );
    json_schema.add_property(
        "proposed_method",
        "string",
        Option::from(String::from("提案手法の詳細な説明．")),
    );
    json_schema.add_property(
        "experiment_results",
        "string",
        Option::from(String::from("実験の結果の詳細な説明．")),
    );
    json_schema.add_property(
        "comparison_with_related_works",
        "string",
        Option::from(String::from("関連研究と比較した場合のこの論文の新規性についての説明．可能な限り既存研究を参照しながら記述すること．")),
    );
    json_schema.add_property(
        "future_works",
        "string",
        Option::from(String::from(
            "未解決の課題および将来の研究の方向性について記述．",
        )),
    );

    openai
        .model_id("gpt-4o-mini")
        .messages(messages)
        .temperature(1.0)
        .response_format(ResponseFormat::new("json_schema", json_schema));

    let response = openai.chat().unwrap();
    println!("{}", response.choices[0].message.content);
}
