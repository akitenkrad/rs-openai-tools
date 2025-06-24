use super::json_schema::*;
use super::*;

#[test]
fn test_build_json_schema_simple() {
    let mut json_schema = JsonSchema::new(String::from("test-schema"));
    json_schema.add_property(String::from("test_property"), String::from("string"), None);

    let schema_string = serde_json::to_string(&json_schema).unwrap();
    println!("{}", serde_json::to_string_pretty(&json_schema).unwrap());

    assert_eq!(
        schema_string,
        r#"{"name":"test-schema","schema":{"type":"object","properties":{"test_property":{"type":"string"}},"required":["test_property"],"additionalProperties":false}}"#
    );
}

#[test]
fn test_build_json_schema_with_description() {
    let mut json_schema = JsonSchema::new(String::from("test-schema"));
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

#[test]
fn test_build_json_schema_add_array() {
    let mut json_schema = JsonSchema::new(String::from("test-schema"));
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

#[test]
fn test_build_body_simple() {
    let body = ChatCompletionRequestBody::new(
        "gpt-4o-mini".to_string(),
        vec![Message::new(
            String::from("assistant"),
            String::from("Hi there! How can I assist you today?"),
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
    let mut json_schema = JsonSchema::new(String::from("test-schema"));
    json_schema.add_property(
        String::from("test_property"),
        String::from("string"),
        Option::from(String::from("This is a test property")),
    );
    let response_format = ResponseFormat::new(String::from("json_schema"), json_schema);
    let body = ChatCompletionRequestBody::new(
        "gpt-4o-mini".to_string(),
        vec![Message::new(
            String::from("assistant"),
            String::from("Hi there! How can I assist you today?"),
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
    let messages = vec![Message::new(
        String::from("user"),
        String::from("Hi there!"),
    )];

    openai
        .model_id(String::from("gpt-4o-mini"))
        .messages(messages)
        .temperature(1.0);

    let response = openai.chat().unwrap();
    println!("{}", &response.choices[0].message.content);
    assert!(true);
}

#[test]
fn test_chat_completion_2() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new(
        String::from("user"),
        String::from("トンネルを抜けると？"),
    )];

    openai
        .model_id(String::from("gpt-4o-mini"))
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
        String::from("user"),
        String::from(
            "Hi there! How's the weather tomorrow in Tokyo? If you can't answer, report error.",
        ),
    )];

    let mut json_schema = JsonSchema::new(String::from("weather"));
    json_schema.add_property(
        String::from("location"),
        String::from("string"),
        Option::from(String::from("The location to check the weather for.")),
    );
    json_schema.add_property(
        String::from("date"),
        String::from("string"),
        Option::from(String::from("The date to check the weather for.")),
    );
    json_schema.add_property(
        String::from("weather"),
        String::from("string"),
        Option::from(String::from("The weather for the location and date.")),
    );
    json_schema.add_property(
        String::from("error"),
        String::from("string"),
        Option::from(String::from(
            "Error message. If there is no error, leave this field empty.",
        )),
    );
    openai
        .model_id(String::from("gpt-4o-mini"))
        .messages(messages)
        .temperature(1.0)
        .response_format(ResponseFormat::new(
            String::from("json_schema"),
            json_schema,
        ));

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
        String::from("user"),
        String::from("Hi there! How's the weather tomorrow in Tokyo? Today is 3024/12/25. If you can't answer, report error."),
    )];

    let mut json_schema = JsonSchema::new(String::from("weather"));
    json_schema.add_property(
        String::from("location"),
        String::from("string"),
        Option::from(String::from("The location to check the weather for.")),
    );
    json_schema.add_property(
        String::from("date"),
        String::from("string"),
        Option::from(String::from("The date to check the weather for.")),
    );
    json_schema.add_property(
        String::from("weather"),
        String::from("string"),
        Option::from(String::from("The weather for the location and date.")),
    );
    json_schema.add_property(
        String::from("error"),
        String::from("string"),
        Option::from(String::from(
            "Error message. If there is no error, leave this field empty.",
        )),
    );
    openai
        .model_id(String::from("gpt-4o-mini"))
        .messages(messages)
        .temperature(1.0)
        .response_format(ResponseFormat::new(
            String::from("json_schema"),
            json_schema,
        ));

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

    let messages = vec![Message::new(String::from("user"), instruction.clone())];

    let mut json_schema = JsonSchema::new(String::from("summary"));
    json_schema.add_property(
        String::from("is_survey"),
        String::from("boolean"),
        Option::from(String::from(
            "この論文がサーベイ論文かどうかをtrue/falseで判定．",
        )),
    );
    json_schema.add_property(
        String::from("research_question"),
        String::from("string"),
        Option::from(String::from("この論文のリサーチクエスチョンの説明．この論文の背景や既存研究との関連も含めて記述する．")),
    );
    json_schema.add_property(
        String::from("contributions"),
        String::from("string"),
        Option::from(String::from(
            "この論文のコントリビューションをリスト形式で記述する．",
        )),
    );
    json_schema.add_property(
        String::from("dataset"),
        String::from("string"),
        Option::from(String::from(
            "この論文で使用されているデータセットをリストアップする．",
        )),
    );
    json_schema.add_property(
        String::from("proposed_method"),
        String::from("string"),
        Option::from(String::from("提案手法の詳細な説明．")),
    );
    json_schema.add_property(
        String::from("experiment_results"),
        String::from("string"),
        Option::from(String::from("実験の結果の詳細な説明．")),
    );
    json_schema.add_property(
        String::from("comparison_with_related_works"),
        String::from("string"),
        Option::from(String::from("関連研究と比較した場合のこの論文の新規性についての説明．可能な限り既存研究を参照しながら記述すること．")),
    );
    json_schema.add_property(
        String::from("future_works"),
        String::from("string"),
        Option::from(String::from(
            "未解決の課題および将来の研究の方向性について記述．",
        )),
    );

    openai
        .model_id(String::from("gpt-4o-mini"))
        .messages(messages)
        .temperature(1.0)
        .response_format(ResponseFormat::new(
            String::from("json_schema"),
            json_schema,
        ));

    let response = openai.chat().unwrap();
    println!("{}", response.choices[0].message.content);
}

#[test]
fn test_chat_completion_with_long_arguments() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new(
        String::from("user"),
        String::from(
            r#"Hi there! How's the weather tomorrow in Tokyo? If you can't answer, report error.
        Context: Tokyo is a bustling metropolis with a rich history and a vibrant culture. It is known for its advanced technology, delicious cuisine, and beautiful parks. The city experiences four distinct seasons, with hot summers and cold winters. The weather can change rapidly, so it's always good to check the forecast before making plans.
        Here is a long context to test the model's ability to handle extended input:
        Tokyo is the capital city of Japan, known for its vibrant culture, advanced technology, and rich history. The city experiences four distinct seasons, with hot, humid summers and mild winters. Weather forecasting in Tokyo is generally reliable, but unexpected changes can occur due to its proximity to the ocean and mountainous regions.
        In recent years, Tokyo has hosted several international events, including the 2020 Summer Olympics. The city's infrastructure is designed to handle large populations and frequent visitors, with efficient public transportation and numerous amenities.
        When considering the weather for tomorrow, it is important to note that forecasts may vary depending on the source. The Japan Meteorological Agency provides up-to-date information, but sudden shifts in atmospheric pressure or typhoons can lead to rapid changes.
        If you are planning outdoor activities in Tokyo, it is advisable to check the weather forecast regularly and prepare for possible rain or strong winds, especially during the typhoon season from August to October.
        Additionally, Tokyo's weather can be influenced by global climate patterns such as El Niño and La Niña, which may cause unusual temperature fluctuations or precipitation levels.
        In summary, while Tokyo generally enjoys a temperate climate, it is always best to stay informed about the latest weather updates to ensure a safe and enjoyable experience in the city.

        Context: The climate in Tokyo is characterized by a humid subtropical climate zone, with significant seasonal variation. Spring brings cherry blossoms and mild temperatures, while summer is hot and humid, often accompanied by heavy rainfall and occasional typhoons. Autumn is generally pleasant with cooler temperatures and clear skies, making it a popular season for outdoor activities. Winter is relatively mild compared to other regions of Japan, with rare snowfall and crisp, dry air. Due to its geographical location, Tokyo can experience sudden weather changes, so residents and visitors are advised to monitor weather reports, especially during the rainy and typhoon seasons.
        Here is some additional long context to further test the model's ability to process extended input:
        Tokyo's urban landscape is a blend of modern skyscrapers and historic temples, reflecting centuries of cultural evolution. The city is divided into several wards, each with its own unique character and attractions. Shibuya is famous for its bustling crossing and youth culture, while Asakusa offers a glimpse into traditional Japan with the iconic Senso-ji Temple. The efficient public transportation system, including the extensive subway and train networks, makes it easy to navigate the city and explore its diverse neighborhoods.
        The weather in Tokyo can have a significant impact on daily life, influencing everything from commuting patterns to seasonal festivals. During the rainy season, known as "tsuyu," residents often carry umbrellas and adjust their schedules to avoid heavy downpours. In contrast, the cherry blossom season in spring draws crowds to parks and riversides, where people gather for "hanami" picnics under the blooming trees.
        Tokyo's proximity to the Pacific Ocean means that it is occasionally affected by typhoons, which can bring strong winds and heavy rainfall. The city's infrastructure is designed to withstand such events, with advanced drainage systems and strict building codes. However, residents are still advised to stay informed through weather alerts and prepare emergency supplies in case of severe storms.
        In recent years, climate change has led to more frequent heatwaves and unpredictable weather patterns in Tokyo. The local government has implemented measures to mitigate the effects of extreme heat, such as installing misting stations and promoting the use of shade structures in public spaces. Environmental awareness campaigns encourage residents to conserve energy and reduce their carbon footprint.
        Overall, Tokyo's dynamic climate and vibrant urban environment make it a fascinating subject for weather-related inquiries and research. Whether planning a visit or studying the city's meteorological trends, staying informed about the latest weather updates is essential for making the most of what Tokyo has to offer.

        Context: Tokyo is a city that never sleeps, with a rich tapestry of culture, technology, and history woven into its fabric. The weather in Tokyo can be as dynamic as the city itself, with each season bringing its own unique charm and challenges. From the cherry blossoms of spring to the vibrant foliage of autumn, the city's climate plays a significant role in shaping the experiences of both residents and visitors.
        The summer months in Tokyo are characterized by high humidity and temperatures that can soar above 30 degrees Celsius. This is also the time when the city experiences its rainy season, known as "tsuyu," which typically lasts from early June to mid-July. During this period, heavy rainfall can lead to localized flooding and transportation disruptions, so it's essential to stay updated on weather forecasts.
        As the summer heat gives way to autumn, Tokyo transforms into a canvas of red and gold as the leaves change color. The cooler temperatures make it an ideal time for outdoor activities, and many festivals celebrate the beauty of the season. However, typhoons can still pose a threat during this time, bringing strong winds and heavy rain.
        Winter in Tokyo is relatively mild compared to other parts of Japan, with temperatures rarely dropping below freezing. Snowfall is infrequent but can occur, creating a picturesque scene in the city's parks and gardens. The dry air and clear skies make it a great time for stargazing and enjoying the city's illuminated landmarks.
        Despite the challenges posed by its climate, Tokyo remains a resilient city that adapts to its weather patterns. From innovative urban planning to community preparedness initiatives, the city's response to its dynamic climate is a testament to its enduring spirit and commitment to sustainability.

        Context: Tokyo is a city that thrives on its dynamic climate, with each season offering a unique experience. The weather can change rapidly, so it's essential to stay informed about the latest forecasts. The city is known for its hot, humid summers and mild winters, with occasional snowfall. Spring brings beautiful cherry blossoms, while autumn showcases stunning foliage. Typhoons can occur during the late summer and early autumn months, bringing heavy rain and strong winds. It's always a good idea to check the weather before planning outdoor activities in Tokyo.
        Tokyo's weather is influenced by its geographical location, surrounded by mountains and the Pacific Ocean. This results in a humid subtropical climate, with distinct seasons that can affect daily life. The city experiences a rainy season in June and July, which can lead to localized flooding. However, the city's infrastructure is well-equipped to handle such events, ensuring minimal disruption to transportation and daily activities.
        When it comes to weather forecasting, Tokyo has a reliable system in place. The Japan Meteorological Agency provides accurate and timely updates, allowing residents and visitors to plan accordingly. Whether you're looking to enjoy a sunny day in one of Tokyo's many parks or need to prepare for a sudden downpour, staying informed about the weather is key to making the most of your time in this vibrant city.
        "#,
        ),
    )];

    openai
        .model_id(String::from("gpt-4o-mini"))
        .messages(messages)
        .temperature(1.0);

    let response = openai.chat().unwrap();
    println!("{:#?}", response);
}
