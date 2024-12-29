[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/X1fiE4koKU88Z9sKwWoPAH/3BzpSDuoSYUk4t6jjGMM5X/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/X1fiE4koKU88Z9sKwWoPAH/3BzpSDuoSYUk4t6jjGMM5X/tree/main)
![Crates.io Version](https://img.shields.io/crates/v/openai-tools?style=flat-square&color=blue)

# OpenAI Tools

API Wrapper for OpenAI API.

<img src="../LOGO.png" alt="LOGO" width="150" height="150">

## Installation

To start using the `openai-tools`, add it to your projects's dependencies in the `Cargo.toml' file:

```bash
cargo add openai-tools
```

API key is necessary to access OpenAI API.  
Set it in the `.env` file:

```text
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Then, import the necesarry modules in your code:

```rust
use openai_tools::OpenAI;
```

## Usage

### Chat Completion

- Simple Chat

    ```rust
    let mut openai = OpenAI::new();
    let messages = vec![
        Message::new("user", "Hi there!")
    ];

    openai
        .model_id("gpt-4o-mini")
        .messages(messages)
        .temperature(1.0);

    let response: Response = openai.chat().unwrap();
    println!("{}", &response.choices[0].message.content);
    // Hello! How can I assist you today?
    ```

- Chat with Json Schema

    ```rust
    #[derive(Debug, Serialize, Deserialize)]
    struct Weather {
        location: String,
        date: String,
        weather: String,
        error: String,
    }

    let mut openai = OpenAI::new();
    let messages = vec![Message::new(
        "user",
        "Hi there! How's the weather tomorrow in Tokyo? If you can't answer, report error."
            ,
    )];

    // build json schema
    let mut json_schema = JsonSchema::new("weather".to_string());
    json_schema.add_property(
        "location".to_string(),
        "string".to_string(),
        Option::from("The location to check the weather for.".to_string()),
    );
    json_schema.add_property(
        "date".to_string(),
        "string".to_string(),
        Option::from("The date to check the weather for.".to_string()),
    );
    json_schema.add_property(
        "weather".to_string(),
        "string".to_string(),
        Option::from("The weather for the location and date.".to_string()),
    );
    json_schema.add_property(
        "error".to_string(),
        "string".to_string(),
        Option::from("Error message. If there is no error, leave this field empty.".to_string()),
    );

    // configure chat completion model
    openai
        .model_id("gpt-4o-mini")
        .messages(messages)
        .temperature(1.0)
        .response_format(ResponseFormat::new("json_schema".to_string(), json_schema));

    // execute chat
    let response = openai.chat().unwrap();

    let answer: Weather = serde_json::from_str::<Weather>(&response.choices[0].message.content)
    println!("{:?}", answer)
    // Weather {
    //     location: "Tokyo",
    //     date: "2023-10-01",
    //     weather: "Temperatures around 25°C with partly cloudy skies and a slight chance of rain.",
    //     error: "",
    // }
    ```
