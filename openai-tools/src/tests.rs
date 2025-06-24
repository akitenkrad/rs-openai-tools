use super::json_schema::*;
use super::*;
use tiktoken_rs::o200k_base;

#[tokio::test]
async fn test_build_json_schema_simple() {
    let mut json_schema = JsonSchema::new(String::from("test-schema"));
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

#[tokio::test]
async fn test_build_json_schema_add_array() {
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

#[tokio::test]
async fn test_build_body_simple() {
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

#[tokio::test]
async fn test_build_body_with_args() {
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

#[tokio::test]
async fn test_deserialize_api_response() {
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

#[tokio::test]
async fn test_chat_completion() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new(
        String::from("user"),
        String::from("Hi there!"),
    )];

    openai
        .model_id(String::from("gpt-4o-mini"))
        .messages(messages)
        .temperature(1.0);

    let response = openai.chat().await.unwrap();
    println!("{}", &response.choices[0].message.content);
    assert!(true);
}

#[tokio::test]
async fn test_chat_completion_2() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new(
        String::from("user"),
        String::from("トンネルを抜けると？"),
    )];

    openai
        .model_id(String::from("gpt-4o-mini"))
        .messages(messages)
        .temperature(1.5);

    let response = openai.chat().await.unwrap();
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

#[tokio::test]
async fn test_chat_completion_with_json_schema() {
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

    let response = openai.chat().await.unwrap();
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

#[tokio::test]
async fn test_chat_completion_with_json_schema_expect_error() {
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

    let response = openai.chat().await.unwrap();
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

#[tokio::test]
async fn test_summarize() {
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

    let response = openai.chat().await.unwrap();
    println!("{}", response.choices[0].message.content);
}

#[tokio::test]
async fn test_chat_completion_with_long_arguments() {
    let mut openai = OpenAI::new();
    let messages = vec![Message::new(
        String::from("user"),
        String::from(
            r#"Prologue: The Breath Before the Trigger
The wind was quiet that morning—too quiet for the season.
A pale sun crawled over the jagged horizon, casting long shadows across the frostbitten valley. The mountains stood like forgotten gods, their shoulders dusted in snow, silent watchers over the world below. Between them lay a sliver of steel and wood—perfectly still, like it had grown from the earth. It was not just a rifle. It was history, vengeance, mathematics, and instinct. It was precision made manifest.
They called it Valkyrie.
Not for its elegance, though it had that. Nor for its legacy, though bloodied scrolls of war would tell tales of its lineage. No, it earned its name for what it summoned: the judgment of the unseen. The last breath. The final beat before silence. For in the right hands, it did not miss.
Long before it came into the possession of its current wielder, Valkyrie had traveled far—from secret factories hidden beneath tundras, through ghost wars fought in deserts no map dared mark, to auction tables in shadowed rooms where generals and ghosts bartered with more than money. Each owner left a part of themselves in the cold grooves of its barrel. Each trigger pull etched their memory into its metal. And still it endured, patient.
Its latest bearer—nameless for now—lay prone on the ridge, the earth damp beneath their chest, their breath slow, matching the rhythm of the grass swaying beside them. They were not new to the craft. They did not worship the weapon. They understood it. The way a poet understands silence. The way a surgeon respects the scalpel.
Below, far below, a convoy moved through the ravine like ants unaware of the boot above. Trucks, men, secrets. There were targets there. Perhaps one. Perhaps many. Perhaps none. The data was uncertain. The order was not.
"Wait for the wind."
They remembered that lesson well. The first rule, passed down from a teacher long vanished: patience is the sniper’s most dangerous caliber. Not the bullet. Not the scope. But the breath before the trigger. That fraction of a moment when time pauses, when fate balances on the edge of a hairline.
And so, they waited.
As the light grew, so did the stillness. Somewhere far away, a raven screamed into the silence, and the sound fell flat against the sky. The rifle remained still. The eye behind the scope watched. Not for movement. But for meaning.
This is not the story of a single shot. Nor is it a tale of war and glory. This is the story of the sniper rifle itself—of those who carried it, those who feared it, and those who never even heard the crack before the world went dark.
This is the story of Valkyrie.
And this is only the beginning.

Chapter 2: Echoes in Brass
Empty shell casings have memories.
Long after the echo fades, after the smoke has drifted into myth, they remain—burnt, hollow, discarded. But they remember. The trembling finger. The exhale. The choice. Some call it the price of duty. Others, the weight of killing. But for those who live by the scope, it is neither burden nor pride. It is necessity.
The shell from this morning’s shot now lay buried in the grass, glinting dully in the half-light. One shot. One hit. No confirmation needed. Valkyrie did not lie. And yet, the figure who pulled the trigger did not move. The mission was not over. It had just begun.
The sniper—codename Ash—stayed low. Always low.
The name was not theirs by birth, but by fire. Years ago, in the hills of a nation no longer on any map, a village had burned because Ash had not pulled the trigger in time. A convoy rolled through. A deal was struck. An atrocity was made. By the time command issued the go-order, the screams had already started.
Ash survived the inquiry. The commanding officer did not. But the name remained. As a reminder. As penance. As armor.
Now, Ash moved silently through the underbrush, every step a practiced prayer. The target—known only as Moth—had not been among the fallen. Intelligence had lied. Or worse, had been paid to lie. Ash suspected both.
The rifle slung across their back groaned slightly, metal against metal. Valkyrie did not like the quiet after a kill. It was a weapon born to hunt. And Moth was prey worth hunting.
In the years since their paths first crossed, Moth had grown into a phantom. No face. No confirmed voice. Only signals, shadows, and aftermaths. Bombings without cause. Operations that vanished from record. Bodies too mutilated for identification. Always followed by the same pattern: a butterfly charm, scorched and melted, left like a calling card.
Ash had once found one lodged in a child’s skull.
That was when the vendetta began.
A mission briefing had never been so personal. This wasn't about politics. Or even justice. This was about understanding why. Why a man—or whatever Moth truly was—would kill to be seen, yet never step into the light. Why a sniper’s bullet never found him. Why he always seemed to know… just a moment before.
Some said Moth could hear the shot before it was fired.
Ash didn’t believe in ghosts.
But they believed in patterns. And there was one emerging now.
The convoy had not carried weapons. It had carried data. Hardened drives. Radiation-shielded cores. Surveillance-grade encryption. Whatever was on them had enough value to warrant decoys, mercenaries, and, most tellingly—sacrifice.
Ash’s scope had captured one frame, just before the impact. A face. Not Moth’s. But someone looking up. Not afraid. Aware.
The kind of look someone gives a camera they didn’t expect to find.
Ash had already pulled the image, burned it into a drive, and left it for dead-drop extraction. The trail was warm. The time to hunt had come.
Valkyrie was cleaned that night. Piece by piece. Ritual by ritual. Each part laid out on cloth like sacred scripture. The action was silent, practiced. Every sniper has their rites, and Ash had theirs: dismantle the weapon to remember the weight of its unity. Oil the bolt to honor the precision. Align the scope to never forget what a single blink can cost.
Then came the final motion: placing the next round into the chamber.
Ash didn’t use full magazines. One bullet. Always one.
Because when the moment comes, there is only ever one shot.
Tomorrow, the chase would resume. But tonight, in the rusted ruins of an abandoned outpost, beneath a moon that refused to be full, Ash whispered to the rifle like a lover:
"We’re not done yet."
And somewhere, hundreds of miles away, a man who went by Moth paused mid-sentence… and smiled.

Chapter 3: The Art of Vanishing
In a nameless city built on secrets and salt, beneath a club that pulsed with manufactured hedonism, Moth lit a cigarette with fingers that had never trembled.
He had been watching the monitors for hours—sixteen screens arranged in perfect angles, each one streaming feeds from the convoy that had failed to reach its destination. Not one showed the face he expected.
Ash.
The sniper had been there. The report was late by forty-three seconds. That was enough. Moth had built empires on smaller discrepancies. The kill had been clean—center mass, wind-compensated, no trace of panic. Ash was improving. Or perhaps evolving.
Moth exhaled slowly, letting the smoke curl toward the ceiling where the air purifiers would devour it like obedient predators. He never smoked in public. Only here, in this sealed room, beneath layers of denial and encryption, did he allow himself that indulgence.
His reflection shimmered faintly in the glass. A face built from memory, not birth. Bone structure altered. Retina replaced. Voice tuned to frequencies that bypassed most lie detectors. There was nothing original left. Moth had murdered his former self long ago.
He hadn't always been a monster.
Once, long before the name Moth clung to him like soot, he had believed in things. Borders. Orders. Right and wrong. The sniper rifle that Ash now carried had once belonged to a man Moth had trained beside—the man, in fact, who had taught them both.
But where Ash took the teachings and buried them like roots, Moth burned them down to ash.
It was ideology that betrayed him. He remembered the tribunal. The fabricated charges. The smirk of the man who pulled the strings. How many nations had he bled for, only to be cast out when the blood dried?
He became invisible not by choice, but by necessity.
Yet invisibility became power.
He learned to build networks of silence. Hired ghosts. Operated in code so subtle it passed for coincidence. He discovered that nothing in this world was more feared than a face without a name.
Ash, though, was the exception.
Ash remembered his name.
That was the problem.
Moth flicked the cigarette into the glass tray and turned toward a display console where a single red dot pulsed near the border of what used to be a war zone. The signal was faint, but consistent.
"She's close," he muttered.
A woman entered the room without knocking—short hair, tactical posture, a pistol holstered upside-down beneath her jacket. Lena. Former SIS, current operations chief. Efficient, ruthless, loyal only to the contract.
“She’s already moved beyond the ridge. We have ten hours at most before she hits the secondary cache.”
Moth tapped the screen, zooming into the signal.
“She doesn’t know what’s in it.”
“She doesn’t need to,” Lena replied. “Only that you want it.”
Moth smiled, cold and deliberate.
“She’ll take the bait.”
“You're sure?” Lena asked. “Ash isn’t like the others.”
“No,” Moth said. “She’s worse.”
He walked to the locker, input a code that changed every hour, and retrieved a weapon unlike any he’d used in years. Short-barreled, suppressed, matte black. Not a sniper’s rifle. A killer’s whisper.
“She’s coming for me,” he said. “Let her.”
Then he looked at the wall, where an old photo—faded, grainy, perhaps deliberately damaged—showed three people in uniform. Two men, one woman. Young. Smiling. One of them held a rifle marked with an old inscription in Cyrillic.
Valkyrie.
Moth tapped the photo once, gently.
“You trained her well, old friend,” he whispered. “Let’s see if she learned everything.”

Chapter 4: No One Hears the Grass Grow
Ash had always preferred the silence before dawn.
It was not the silence of absence but of possibility. Before the first bird dared to sing, before the machines began their endless hum, the world paused—just long enough for a hunter to listen. And right now, that silence spoke volumes.
Ash crouched beside a stream no wider than a boot. The water was slow, cautious—like it, too, sensed something unnatural in the air. Across the clearing, beneath a bluff wrapped in brittle moss and shattered roots, lay the cache. Not buried. Not hidden. Just placed—as if someone wanted it found.
A trap.
Ash’s breath moved like smoke through cloth. The ghillie suit fluttered faintly, blending with the reeds. Every inch of their movement was calculated: heel to toe, muscle to moss. The rifle, cradled like an extension of the spine, remained still.
Valkyrie hadn’t spoken yet.
Ash always imagined the rifle had moods—not voices, not commands, but instincts. When something was wrong, the scope would fog slightly, or the bolt would catch half a breath longer than expected. This morning, Valkyrie was nervous.
Still, Ash moved.
It took 47 minutes to cross 600 meters. Every approach path had already been plotted the night before. Every exit route, memorized. There was no backup. No drone. No failsafe.
Ash trusted no one.
A footfall away from the cache, something shifted.
Not in the landscape. In the pattern.
A thread pulled loose in the air.
Ash’s eyes locked onto a glint—imperceptible to most, but to a sniper’s trained gaze, it screamed. A fiber-optic filament, broken at the tip, likely feeding light to a passive sensor. Not amateur work. Industrial-grade camouflage.
Moth was watching.
Ash didn’t flinch. Didn’t blink. Instead, in one motion so fluid it felt rehearsed, they dropped flat, rolled sideways, and aimed upward—not at the cache, but at the outcropping above and behind it.
There.
Just for an instant: movement.
A flicker of fabric. The shimmer of optics.
Ash didn’t shoot.
Not yet.
Instead, they waited. A second. Two. A shadow crossed the bluff, but it didn’t run. It circled. Professional. Not Moth—he didn’t do his own dirty work anymore. But one of his new ghosts, perhaps. Ash would test them.
A distraction grenade rolled from the side pack. Silent fuse. No light, no sound—just a pressure wave designed to rupture equilibrium. Ash launched it toward the base of the bluff, then moved—fast, low, toward the stream, away from the cache.
The detonation was muted. A bird shrieked. No gunfire.
Ash counted: One… Two… Three…
And then the rifle barked.
Not Valkyrie. Something smaller. Suppressed.
The shot hit the tree behind them—wrong angle, off mark.
Amateur mistake.
Ash pivoted, dropped into firing stance, and Valkyrie sang.
A single shot. No echo. No hesitation.
The bluff returned to silence.
Ash waited for movement. None came. Then rose and moved—swift and ghostlike—to the source. There, amid torn ferns and blood-slicked stone, lay a body in tactical black. No insignia. No dog tag. Just a mark on the glove: a stylized moth etched in silver.
Ash knelt, turned the head gently. Young. Probably mercenary. Indoctrinated, not trained. There would be more.
The cache was never the goal.
It was the invitation.
Ash looked skyward. The sun was barely rising, but it already felt late.
Moth was drawing closer. No more intermediaries. No more games. The field was narrowing.
Ash whispered, “I’m coming.”
Valkyrie remained silent. But it felt like approval.

Chapter 5: Before the Names
Five years ago, the world still made sense—at least on paper.
There was no Ash then. No Moth. Just two operatives in the service of a program that officially didn’t exist: Project Halcyon, a deniable initiative buried within the intelligence branches of six nations, designed to train and deploy off-the-record snipers to places where flags dared not fly.
They called themselves The Silent Three.
Alexei Morozov, the mentor—ex-Spetsnaz, with eyes that saw through lies like glass. The one who taught them how to slow their heartbeat, how to disappear behind the wind, how to kill without hating.
Second, there was her—young, precise, calculating. Back then, her name was Seren Ward. A former military analyst with perfect visual recall and a surgeon’s patience. They said she could hit a moving target at 1,000 meters and forget she ever pulled the trigger.
And the third…
He was charming then. Always one step ahead. They knew him as Victor Delane. The instructors called him the ghost that smiles. He had a gift—not just for shooting, but for slipping into roles, into accents, into people’s trust. He could kill and make you thank him for it.
Seren and Victor were a perfect match.
Not romantically—though there had been moments, late at night, whispers that never turned into kisses. But tactically. Intellectually. They finished each other’s calibrations. Anticipated each other’s firing positions without speaking. They moved like mirrored thoughts.
Alexei pushed them hard, but not cruelly. He saw something rare: two souls shaped by war, but unbroken by it. He told them once, over tea and silence, that one day the wind would change—and that when it did, they would either hold each other up or destroy each other completely.
He was right.
It happened in the desert.
A black op so sensitive it never made the logs. A warlord in possession of a data payload—encrypted files that hinted at corruption not just in foreign regimes, but in their own chain of command. Halcyon was ordered to retrieve the payload. No witnesses. No survivors.
Seren hesitated.
She saw the faces of children in the encampment. They weren’t just collateral. They were the bargain. The price for silence. She broke protocol. Warned the locals. Extracted the data, but left the warlord alive. She thought Victor had her back.
She was wrong.
Victor filed the report. Seren was labeled a liability.
Alexei disappeared a week later—officially “killed in an ambush,” but Seren had always suspected something else. Victor… simply vanished. Not discharged. Not reprimanded. Transferred—to where, no one said.
It wasn’t until a year later, during a recon mission in Eastern Europe, that she saw the calling card for the first time:
A butterfly charm.
Charred at the edges.
She didn’t need a name to know who had left it.
Victor had become Moth.
And Seren shed her own name like dead skin.
From then on, she was only Ash.
A new ghost for a new war.
Ash woke from the memory with a start.
The campfire had gone cold. Morning mist curled around the rocks like smoke from a distant fire. In her hands, Valkyrie felt heavy—not with weight, but with memory.
The chase was no longer professional. It hadn’t been for years.
It was personal now.
She didn’t hunt Moth because of orders. She hunted him because she had loved who he used to be—and mourned who he became.
And she wasn’t sure which grief was heavier.
Ash stood, chambered a round, and looked east.
The wind was changing.

Chapter 6: A Smile in the Scope
He hadn’t dreamed of her in years.
But last night, she returned—not as Ash, not as an adversary cloaked in shadows and vengeance, but as Seren. In the dream, they stood on the edge of a ruined watchtower, wind howling around them, maps clutched in their hands. Her eyes were sharp, her expression unreadable. She said nothing. Just looked at him. And he knew what the silence meant.
“You left before I asked you to.”
Moth awoke to the bitter taste of metal. Not blood—regret. It was always sharper than anything forged.
He sat up in the armory chamber of one of his satellite compounds—a monolithic shelter buried beneath a forgotten railway station in the Baltics. Concrete walls bled with condensation, and rows of weaponry stared back at him like old accusations. Everything here had a story. And every story pointed to one conclusion:
Ash was close.
Lena stood by the console when he emerged. She didn’t flinch at his sudden presence.
“She neutralized the bluff team,” she said without turning.
“Expected,” Moth replied. “They were just the opening act.”
“Then why send them?”
“To remind her this is still a performance.”
He walked past her and tapped into the comms terminal. A real-time satellite feed spun into motion—infrared sweeps, radio packet sniffers, deep acoustic surveillance grids. He traced Ash’s path by the absence of disruption. Where others stumbled, she moved like a shadow between atoms.
“She’ll go for the cache in Sector 9 next,” Lena offered.
“She’ll skip it,” Moth corrected. “She’ll assume it’s a diversion.”
“It is a diversion.”
“Exactly.”
Lena finally turned. “You want her to come here.”
Moth smiled—though it barely reached his eyes.
“I need her to.”
He paused at the old locker—the one he hadn’t opened in years. Not since Halcyon fell.
Inside, preserved in foam and memory, was his first sniper rifle.
It was nothing like Valkyrie. Less elegant. Less forgiving. But it had history. He’d once let Seren hold it. She had called it ugly, but she smiled when she said it. That was rare for her. Back then, her smiles were currency—precious, valuable, withheld until earned.
Moth picked up the rifle and held it to his shoulder.
The weight was wrong.
Or maybe he was.
He set it down and opened the side compartment instead, retrieving a small, thin box. Inside lay a photo, untouched by time—faded only at the corners. Three people. A field. Laughter caught mid-breath.
Alexei. Seren. Himself.
None of them had names then.
Just codes. Just belief.
“I didn’t leave you,” he said quietly, to no one. “I saved you.”
Lena said nothing. She had heard these half-confessions before.
Moth closed the box and turned to face the map on the far wall—an embossed topographical relief of the coming battlefield.
The location had been chosen carefully.
Old ruins. Choke points. Elevated sightlines.
But also… memory.
It was where Halcyon had begun its field trials.
Where Seren had taken her first confirmed shot.
Where everything that mattered was born—and where, if fate allowed it, everything would end.
“She’ll come,” Moth said, voice low.
“And if she doesn’t?” Lena asked.
Moth stepped into the light. The years had aged him differently than most. His eyes had grown colder, but his pulse had not slowed.
“She will. Because she’s not chasing me.”
He looked at his own reflection in the blackened monitor.
“She’s chasing who I used to be.”

Chapter 7: The Weight of a Single Shot
Ash moved like water through the ruins.
Each step was a negotiation with time—quiet, patient, exact. The old Halcyon training grounds had long since been reclaimed by moss and silence. Cracked stone courtyards where gunfire once echoed were now blanketed in fog. Firing ranges turned to graveyards. Buildings collapsed inward, like memories trying to forget themselves.
But she remembered.
Every wall.
Every window.
Every line of fire.
Ash crouched beneath what had once been the overwatch tower—her old nest. The wind hadn’t changed. Not yet. But the stillness had. There was a presence in the air, like something watching from behind the veil.
Moth was near.
She didn’t need proof. She felt it—like gravity pulling at her ribs.
She set down her pack. Unclipped Valkyrie. Began the ritual.
The rifle came apart not out of need, but memory. She checked every part. Ran her fingers along the bolt. Wiped the optics. Whispered to the scope like it might respond. This rifle had taken lives, yes—but more than that, it had witnessed.
Each target. Each moment. Each choice she could never take back.
She laid out her final three bullets.
One was marked with a red line.
His name.
If it came to it, she would use that one last.
Ash checked her watch.
3 hours until first light.
Moth liked the dawn. He had once said that a sniper’s best ally was contradiction—kill in the hour people felt safest. Shoot just as the birds sing. Make the world doubt itself.
She had doubted herself enough for a lifetime.
Ash reached into her pack and pulled out the old photo.
The one she shouldn’t have kept.
Alexei. Victor. Herself.
They were so young. She had kept her hair short even then—too practical for vanity. Victor had that damned smirk—the one that said he knew five things you didn’t, and one of them could save your life.
She hadn’t smiled in that photo. But she remembered the moment it was taken.
They had just completed the “Final Calibration” exercise—snipers paired off, given conflicting orders, forced to choose whether to complete the mission or trust their partner’s instinct. It was a test of loyalty. And morality. And command.
She and Victor had both disobeyed.
They had chosen each other.
That night, Alexei bought them vodka and said nothing.
The silence was his approval.
Ash tucked the photo away. Looked to the sky.
Still no stars.
Too much cloud cover. That was good. She preferred cover to clarity.
In the distance, a soft click echoed.
Too faint for most.
Not for her.
She froze. Heart slowed. Focus sharpened.
A footfall. A weapon being braced. Not close—maybe 300 meters. Someone setting up, not attacking. Moth, or one of his ghosts?
It didn’t matter.
She pulled Valkyrie back together, smooth as breath. Chambered the red-marked round. Clicked the safety off. Slid prone across the cold concrete, her eyes behind glass, her breath steady.
And there he was.
Through the scope. Not a shadow. Not a decoy.
Him.
Older. Leaner. But still Victor in all the ways that haunted.
Standing atop the ruins of what was once Alexei’s office, framed by a shattered skylight. Looking right at her. No scope. No weapon in hand.
Just watching.
Ash didn’t pull the trigger.
Not yet.
Because he raised a hand.
Not in surrender.
But in recognition.
The same way he used to signal: Trust me. Not yet.
The wind shifted.
And in that moment, Ash realized—
This wasn't just the endgame.
It was the beginning of the truth.

Chapter 8: Cathedrals of Concrete and Ghosts
The ruins were sacred in their own way.
Not to any god, but to a discipline—angles, range, line of sight. Every crack in the wall was a margin of error; every hallway a tunnel of fate. Ash moved through it like a spirit retracing old steps, her body a map of training scars and muscle memory.
Moth had vanished.
The rooftop where he had stood moments ago was now empty—no shell casings, no sensor traces, no heat signatures. Just absence. Intentional. A message.
This isn't your kill yet.
Ash lowered Valkyrie. Her hands were steady. Her heart was not.
She moved deeper into the complex, weaving through collapsed corridors and shattered frames. Her mind stayed in the moment, but her memory kept flickering—images surfacing like bubbles from deep water.
She remembered learning to shoot here.
Alexei standing behind her, pressing a hand to her shoulder. “You don’t pull the trigger,” he’d said. “You release it. Like letting go of something you love.”
She remembered Victor adjusting her aim, teasing her under breath. “You’re aiming like a tactician, Seren. Try aiming like someone who’s already forgiven herself.”
That was the cruelest part.
He had never hated her. Not even in the end.
And yet, he betrayed her all the same.
Ash froze at a stairwell—half-gutted, sagging inward. She scanned the angles. No movement. But something felt wrong.
Then came the voice.
Not loud. Not amplified. Just there. Carried on the still air like an old whisper.
“You still lead with your left foot. I told you it echoes.”
Ash turned, fast. Valkyrie up.
Nothing.
No heat signature. No silhouette. Just static bleeding into her earpiece.
“You shouldn’t be here, Seren.”
Her name. Her real name. No one had spoken it in years.
She kept moving. Slow. Rifle tracking every corridor like a pendulum of fate.
“You think you’re here to kill me. But you're not. Not really.”
The voice wasn’t taunting. It wasn’t even confident.
It was tired.
Ash reached the center chamber—the old operations control room. Roof half gone. Consoles gutted. Moss blooming in the corners. She swept the space. Nothing.
Then: a flutter.
A butterfly charm.
Dangling from a bent aerial, spinning slowly.
Below it, a satchel.
She moved forward, wary. No wires. No traps. Just… a gift.
Inside: an old field recorder. Still warm.
She played it.
“There were three of us. You remember. You, me, Alexei. The last time we stood here, we were gods. At least, we thought we were.”
“I betrayed you. Yes. But I didn’t kill you. That was your choice. You killed Seren. I only helped you become Ash.”
Ash stood frozen. Her jaw clenched. Her grip tightened.
“And now you're here to erase the last piece of your past.”
“But if you do, you'll never know why.”
The recording clicked off.
No coordinates. No riddles.
Just that word—why—dangling like the charm above it.
Ash exhaled. Not because she was calm, but because she realized something:
She wasn’t the one hunting anymore.
Not entirely.
This was his labyrinth now.
But she knew how to break walls.
She lifted Valkyrie.
And whispered to the ghost in her earpiece, “Run all you want. But I still remember how you breathe.”
Then she stepped into the lightless corridor ahead—toward the heart of the ruins, where the final truth waited.
And perhaps, the shot that would end both of them.

Chapter 9: The Distance Between Crosshairs
Ash moved deeper into the ruins.
Each step echoed not in the air, but in her spine. Her body remembered this place—where her hands first learned to steady a rifle, where her trust had once felt indestructible, where everything began to fracture.
The corridor narrowed, leading into the inner sanctum of the old Halcyon complex. Once, this room had held servers, surveillance boards, orders never written. Now it was hollow. Except for the two of them.
Moth stood at the far end.
Not hiding. Not aiming.
Waiting.
Ash entered silently, Valkyrie half-raised, not pointed.
He didn't flinch.
The space between them was thirty meters. A sniper’s insult of a distance—too close for comfort, too far for trust. And yet neither moved.
For a long moment, the only sound was the wind threading through the broken ceiling above them.
Ash broke it first.
"Why here?"
Moth didn’t smile. For once.
“Because this is where we stopped being real.”
Ash's finger rested on the trigger guard. Not threatening. Prepared.
“You could’ve killed me ten times over,” she said.
“I still could.”
“Then why didn’t you?”
He looked away. Not in fear, but in thought.
“Because you’re the only person who might understand.”
Ash’s eyes narrowed. The scar under her left brow twitched—the one from the shrapnel Victor once pulled out, years ago, with shaking hands and teeth clenched against the pain.
“Understand what?”
Moth stepped forward, just one pace.
“What Halcyon really was. What Alexei knew. Why he died.”
Silence cracked like glass.
Ash stepped forward too, mirroring him.
“He died in an ambush.”
“That’s the story they sold you.”
He reached into his coat—slowly, deliberately—and pulled out a sealed folder. Tossed it across the floor. It skidded to her boots.
“What’s this?”
“Alexei’s last debrief. He left it for you. But they intercepted it. I stole it back. Took me two years.”
Ash didn’t pick it up. Not yet.
“Why give it to me now?”
Moth's voice dropped.
“Because you need to know what we were built for. Not the wars we fought, but the lies we protected.”
Ash finally lowered Valkyrie.
Just slightly.
“And what were you built for?” she asked, her voice suddenly colder.
Moth looked her in the eye.
And for the first time, Victor looked back.
“To disappear. So people like you could stay whole.”
They stood in silence again—two ghosts in a cathedral of ruin. The distance between them measured in choices, not meters.
Ash knelt. Picked up the folder.
Hands steady.
She opened it.
Her breath caught.
Photographs. Codenames. Black sites she had never been told about. Orders issued in Alexei’s name after his supposed death. Surveillance footage of her—before she was recruited. Her family.
Her real file.
She looked up, eyes burning.
“Why didn’t you tell me?”
“Because you weren’t ready.”
“And now I am?”
Moth nodded once.
“Because if you pull that trigger now, you’ll know exactly what you’re killing.”
Ash looked at Valkyrie.
Then at him.
Then… she turned and walked away.
Not out of mercy.
Not out of fear.
But because the bullet no longer belonged to her.
Not yet.

Prologue: The Breath Before the Trigger
The wind was quiet that morning—too quiet for the season.
A pale sun crawled over the jagged horizon, casting long shadows across the frostbitten valley. The mountains stood like forgotten gods, their shoulders dusted in snow, silent watchers over the world below. Between them lay a sliver of steel and wood—perfectly still, like it had grown from the earth. It was not just a rifle. It was history, vengeance, mathematics, and instinct. It was precision made manifest.
They called it Valkyrie.
Not for its elegance, though it had that. Nor for its legacy, though bloodied scrolls of war would tell tales of its lineage. No, it earned its name for what it summoned: the judgment of the unseen. The last breath. The final beat before silence. For in the right hands, it did not miss.
Long before it came into the possession of its current wielder, Valkyrie had traveled far—from secret factories hidden beneath tundras, through ghost wars fought in deserts no map dared mark, to auction tables in shadowed rooms where generals and ghosts bartered with more than money. Each owner left a part of themselves in the cold grooves of its barrel. Each trigger pull etched their memory into its metal. And still it endured, patient.
Its latest bearer—nameless for now—lay prone on the ridge, the earth damp beneath their chest, their breath slow, matching the rhythm of the grass swaying beside them. They were not new to the craft. They did not worship the weapon. They understood it. The way a poet understands silence. The way a surgeon respects the scalpel.
Below, far below, a convoy moved through the ravine like ants unaware of the boot above. Trucks, men, secrets. There were targets there. Perhaps one. Perhaps many. Perhaps none. The data was uncertain. The order was not.
"Wait for the wind."
They remembered that lesson well. The first rule, passed down from a teacher long vanished: patience is the sniper’s most dangerous caliber. Not the bullet. Not the scope. But the breath before the trigger. That fraction of a moment when time pauses, when fate balances on the edge of a hairline.
And so, they waited.
As the light grew, so did the stillness. Somewhere far away, a raven screamed into the silence, and the sound fell flat against the sky. The rifle remained still. The eye behind the scope watched. Not for movement. But for meaning.
This is not the story of a single shot. Nor is it a tale of war and glory. This is the story of the sniper rifle itself—of those who carried it, those who feared it, and those who never even heard the crack before the world went dark.
This is the story of Valkyrie.
And this is only the beginning.

Chapter 2: Echoes in Brass
Empty shell casings have memories.
Long after the echo fades, after the smoke has drifted into myth, they remain—burnt, hollow, discarded. But they remember. The trembling finger. The exhale. The choice. Some call it the price of duty. Others, the weight of killing. But for those who live by the scope, it is neither burden nor pride. It is necessity.
The shell from this morning’s shot now lay buried in the grass, glinting dully in the half-light. One shot. One hit. No confirmation needed. Valkyrie did not lie. And yet, the figure who pulled the trigger did not move. The mission was not over. It had just begun.
The sniper—codename Ash—stayed low. Always low.
The name was not theirs by birth, but by fire. Years ago, in the hills of a nation no longer on any map, a village had burned because Ash had not pulled the trigger in time. A convoy rolled through. A deal was struck. An atrocity was made. By the time command issued the go-order, the screams had already started.
Ash survived the inquiry. The commanding officer did not. But the name remained. As a reminder. As penance. As armor.
Now, Ash moved silently through the underbrush, every step a practiced prayer. The target—known only as Moth—had not been among the fallen. Intelligence had lied. Or worse, had been paid to lie. Ash suspected both.
The rifle slung across their back groaned slightly, metal against metal. Valkyrie did not like the quiet after a kill. It was a weapon born to hunt. And Moth was prey worth hunting.
In the years since their paths first crossed, Moth had grown into a phantom. No face. No confirmed voice. Only signals, shadows, and aftermaths. Bombings without cause. Operations that vanished from record. Bodies too mutilated for identification. Always followed by the same pattern: a butterfly charm, scorched and melted, left like a calling card.
Ash had once found one lodged in a child’s skull.
That was when the vendetta began.
A mission briefing had never been so personal. This wasn't about politics. Or even justice. This was about understanding why. Why a man—or whatever Moth truly was—would kill to be seen, yet never step into the light. Why a sniper’s bullet never found him. Why he always seemed to know… just a moment before.
Some said Moth could hear the shot before it was fired.
Ash didn’t believe in ghosts.
But they believed in patterns. And there was one emerging now.
The convoy had not carried weapons. It had carried data. Hardened drives. Radiation-shielded cores. Surveillance-grade encryption. Whatever was on them had enough value to warrant decoys, mercenaries, and, most tellingly—sacrifice.
Ash’s scope had captured one frame, just before the impact. A face. Not Moth’s. But someone looking up. Not afraid. Aware.
The kind of look someone gives a camera they didn’t expect to find.
Ash had already pulled the image, burned it into a drive, and left it for dead-drop extraction. The trail was warm. The time to hunt had come.
Valkyrie was cleaned that night. Piece by piece. Ritual by ritual. Each part laid out on cloth like sacred scripture. The action was silent, practiced. Every sniper has their rites, and Ash had theirs: dismantle the weapon to remember the weight of its unity. Oil the bolt to honor the precision. Align the scope to never forget what a single blink can cost.
Then came the final motion: placing the next round into the chamber.
Ash didn’t use full magazines. One bullet. Always one.
Because when the moment comes, there is only ever one shot.
Tomorrow, the chase would resume. But tonight, in the rusted ruins of an abandoned outpost, beneath a moon that refused to be full, Ash whispered to the rifle like a lover:
"We’re not done yet."
And somewhere, hundreds of miles away, a man who went by Moth paused mid-sentence… and smiled.

Chapter 3: The Art of Vanishing
In a nameless city built on secrets and salt, beneath a club that pulsed with manufactured hedonism, Moth lit a cigarette with fingers that had never trembled.
He had been watching the monitors for hours—sixteen screens arranged in perfect angles, each one streaming feeds from the convoy that had failed to reach its destination. Not one showed the face he expected.
Ash.
The sniper had been there. The report was late by forty-three seconds. That was enough. Moth had built empires on smaller discrepancies. The kill had been clean—center mass, wind-compensated, no trace of panic. Ash was improving. Or perhaps evolving.
Moth exhaled slowly, letting the smoke curl toward the ceiling where the air purifiers would devour it like obedient predators. He never smoked in public. Only here, in this sealed room, beneath layers of denial and encryption, did he allow himself that indulgence.
His reflection shimmered faintly in the glass. A face built from memory, not birth. Bone structure altered. Retina replaced. Voice tuned to frequencies that bypassed most lie detectors. There was nothing original left. Moth had murdered his former self long ago.
He hadn't always been a monster.
Once, long before the name Moth clung to him like soot, he had believed in things. Borders. Orders. Right and wrong. The sniper rifle that Ash now carried had once belonged to a man Moth had trained beside—the man, in fact, who had taught them both.
But where Ash took the teachings and buried them like roots, Moth burned them down to ash.
It was ideology that betrayed him. He remembered the tribunal. The fabricated charges. The smirk of the man who pulled the strings. How many nations had he bled for, only to be cast out when the blood dried?
He became invisible not by choice, but by necessity.
Yet invisibility became power.
He learned to build networks of silence. Hired ghosts. Operated in code so subtle it passed for coincidence. He discovered that nothing in this world was more feared than a face without a name.
Ash, though, was the exception.
Ash remembered his name.
That was the problem.
Moth flicked the cigarette into the glass tray and turned toward a display console where a single red dot pulsed near the border of what used to be a war zone. The signal was faint, but consistent.
"She's close," he muttered.
A woman entered the room without knocking—short hair, tactical posture, a pistol holstered upside-down beneath her jacket. Lena. Former SIS, current operations chief. Efficient, ruthless, loyal only to the contract.
“She’s already moved beyond the ridge. We have ten hours at most before she hits the secondary cache.”
Moth tapped the screen, zooming into the signal.
“She doesn’t know what’s in it.”
“She doesn’t need to,” Lena replied. “Only that you want it.”
Moth smiled, cold and deliberate.
“She’ll take the bait.”
“You're sure?” Lena asked. “Ash isn’t like the others.”
“No,” Moth said. “She’s worse.”
He walked to the locker, input a code that changed every hour, and retrieved a weapon unlike any he’d used in years. Short-barreled, suppressed, matte black. Not a sniper’s rifle. A killer’s whisper.
“She’s coming for me,” he said. “Let her.”
Then he looked at the wall, where an old photo—faded, grainy, perhaps deliberately damaged—showed three people in uniform. Two men, one woman. Young. Smiling. One of them held a rifle marked with an old inscription in Cyrillic.
Valkyrie.
Moth tapped the photo once, gently.
“You trained her well, old friend,” he whispered. “Let’s see if she learned everything.”

Chapter 4: No One Hears the Grass Grow
Ash had always preferred the silence before dawn.
It was not the silence of absence but of possibility. Before the first bird dared to sing, before the machines began their endless hum, the world paused—just long enough for a hunter to listen. And right now, that silence spoke volumes.
Ash crouched beside a stream no wider than a boot. The water was slow, cautious—like it, too, sensed something unnatural in the air. Across the clearing, beneath a bluff wrapped in brittle moss and shattered roots, lay the cache. Not buried. Not hidden. Just placed—as if someone wanted it found.
A trap.
Ash’s breath moved like smoke through cloth. The ghillie suit fluttered faintly, blending with the reeds. Every inch of their movement was calculated: heel to toe, muscle to moss. The rifle, cradled like an extension of the spine, remained still.
Valkyrie hadn’t spoken yet.
Ash always imagined the rifle had moods—not voices, not commands, but instincts. When something was wrong, the scope would fog slightly, or the bolt would catch half a breath longer than expected. This morning, Valkyrie was nervous.
Still, Ash moved.
It took 47 minutes to cross 600 meters. Every approach path had already been plotted the night before. Every exit route, memorized. There was no backup. No drone. No failsafe.
Ash trusted no one.
A footfall away from the cache, something shifted.
Not in the landscape. In the pattern.
A thread pulled loose in the air.
Ash’s eyes locked onto a glint—imperceptible to most, but to a sniper’s trained gaze, it screamed. A fiber-optic filament, broken at the tip, likely feeding light to a passive sensor. Not amateur work. Industrial-grade camouflage.
Moth was watching.
Ash didn’t flinch. Didn’t blink. Instead, in one motion so fluid it felt rehearsed, they dropped flat, rolled sideways, and aimed upward—not at the cache, but at the outcropping above and behind it.
There.
Just for an instant: movement.
A flicker of fabric. The shimmer of optics.
Ash didn’t shoot.
Not yet.
Instead, they waited. A second. Two. A shadow crossed the bluff, but it didn’t run. It circled. Professional. Not Moth—he didn’t do his own dirty work anymore. But one of his new ghosts, perhaps. Ash would test them.
A distraction grenade rolled from the side pack. Silent fuse. No light, no sound—just a pressure wave designed to rupture equilibrium. Ash launched it toward the base of the bluff, then moved—fast, low, toward the stream, away from the cache.
The detonation was muted. A bird shrieked. No gunfire.
Ash counted: One… Two… Three…
And then the rifle barked.
Not Valkyrie. Something smaller. Suppressed.
The shot hit the tree behind them—wrong angle, off mark.
Amateur mistake.
Ash pivoted, dropped into firing stance, and Valkyrie sang.
A single shot. No echo. No hesitation.
The bluff returned to silence.
Ash waited for movement. None came. Then rose and moved—swift and ghostlike—to the source. There, amid torn ferns and blood-slicked stone, lay a body in tactical black. No insignia. No dog tag. Just a mark on the glove: a stylized moth etched in silver.
Ash knelt, turned the head gently. Young. Probably mercenary. Indoctrinated, not trained. There would be more.
The cache was never the goal.
It was the invitation.
Ash looked skyward. The sun was barely rising, but it already felt late.
Moth was drawing closer. No more intermediaries. No more games. The field was narrowing.
Ash whispered, “I’m coming.”
Valkyrie remained silent. But it felt like approval.

Chapter 5: Before the Names
Five years ago, the world still made sense—at least on paper.
There was no Ash then. No Moth. Just two operatives in the service of a program that officially didn’t exist: Project Halcyon, a deniable initiative buried within the intelligence branches of six nations, designed to train and deploy off-the-record snipers to places where flags dared not fly.
They called themselves The Silent Three.
Alexei Morozov, the mentor—ex-Spetsnaz, with eyes that saw through lies like glass. The one who taught them how to slow their heartbeat, how to disappear behind the wind, how to kill without hating.
Second, there was her—young, precise, calculating. Back then, her name was Seren Ward. A former military analyst with perfect visual recall and a surgeon’s patience. They said she could hit a moving target at 1,000 meters and forget she ever pulled the trigger.
And the third…
He was charming then. Always one step ahead. They knew him as Victor Delane. The instructors called him the ghost that smiles. He had a gift—not just for shooting, but for slipping into roles, into accents, into people’s trust. He could kill and make you thank him for it.
Seren and Victor were a perfect match.
Not romantically—though there had been moments, late at night, whispers that never turned into kisses. But tactically. Intellectually. They finished each other’s calibrations. Anticipated each other’s firing positions without speaking. They moved like mirrored thoughts.
Alexei pushed them hard, but not cruelly. He saw something rare: two souls shaped by war, but unbroken by it. He told them once, over tea and silence, that one day the wind would change—and that when it did, they would either hold each other up or destroy each other completely.
He was right.
It happened in the desert.
A black op so sensitive it never made the logs. A warlord in possession of a data payload—encrypted files that hinted at corruption not just in foreign regimes, but in their own chain of command. Halcyon was ordered to retrieve the payload. No witnesses. No survivors.
Seren hesitated.
She saw the faces of children in the encampment. They weren’t just collateral. They were the bargain. The price for silence. She broke protocol. Warned the locals. Extracted the data, but left the warlord alive. She thought Victor had her back.
She was wrong.
Victor filed the report. Seren was labeled a liability.
Alexei disappeared a week later—officially “killed in an ambush,” but Seren had always suspected something else. Victor… simply vanished. Not discharged. Not reprimanded. Transferred—to where, no one said.
It wasn’t until a year later, during a recon mission in Eastern Europe, that she saw the calling card for the first time:
A butterfly charm.
Charred at the edges.
She didn’t need a name to know who had left it.
Victor had become Moth.
And Seren shed her own name like dead skin.
From then on, she was only Ash.
A new ghost for a new war.
Ash woke from the memory with a start.
The campfire had gone cold. Morning mist curled around the rocks like smoke from a distant fire. In her hands, Valkyrie felt heavy—not with weight, but with memory.
The chase was no longer professional. It hadn’t been for years.
It was personal now.
She didn’t hunt Moth because of orders. She hunted him because she had loved who he used to be—and mourned who he became.
And she wasn’t sure which grief was heavier.
Ash stood, chambered a round, and looked east.
The wind was changing.

Chapter 6: A Smile in the Scope
He hadn’t dreamed of her in years.
But last night, she returned—not as Ash, not as an adversary cloaked in shadows and vengeance, but as Seren. In the dream, they stood on the edge of a ruined watchtower, wind howling around them, maps clutched in their hands. Her eyes were sharp, her expression unreadable. She said nothing. Just looked at him. And he knew what the silence meant.
“You left before I asked you to.”
Moth awoke to the bitter taste of metal. Not blood—regret. It was always sharper than anything forged.
He sat up in the armory chamber of one of his satellite compounds—a monolithic shelter buried beneath a forgotten railway station in the Baltics. Concrete walls bled with condensation, and rows of weaponry stared back at him like old accusations. Everything here had a story. And every story pointed to one conclusion:
Ash was close.
Lena stood by the console when he emerged. She didn’t flinch at his sudden presence.
“She neutralized the bluff team,” she said without turning.
“Expected,” Moth replied. “They were just the opening act.”
“Then why send them?”
“To remind her this is still a performance.”
He walked past her and tapped into the comms terminal. A real-time satellite feed spun into motion—infrared sweeps, radio packet sniffers, deep acoustic surveillance grids. He traced Ash’s path by the absence of disruption. Where others stumbled, she moved like a shadow between atoms.
“She’ll go for the cache in Sector 9 next,” Lena offered.
“She’ll skip it,” Moth corrected. “She’ll assume it’s a diversion.”
“It is a diversion.”
“Exactly.”
Lena finally turned. “You want her to come here.”
Moth smiled—though it barely reached his eyes.
“I need her to.”
He paused at the old locker—the one he hadn’t opened in years. Not since Halcyon fell.
Inside, preserved in foam and memory, was his first sniper rifle.
It was nothing like Valkyrie. Less elegant. Less forgiving. But it had history. He’d once let Seren hold it. She had called it ugly, but she smiled when she said it. That was rare for her. Back then, her smiles were currency—precious, valuable, withheld until earned.
Moth picked up the rifle and held it to his shoulder.
The weight was wrong.
Or maybe he was.
He set it down and opened the side compartment instead, retrieving a small, thin box. Inside lay a photo, untouched by time—faded only at the corners. Three people. A field. Laughter caught mid-breath.
Alexei. Seren. Himself.
None of them had names then.
Just codes. Just belief.
“I didn’t leave you,” he said quietly, to no one. “I saved you.”
Lena said nothing. She had heard these half-confessions before.
Moth closed the box and turned to face the map on the far wall—an embossed topographical relief of the coming battlefield.
The location had been chosen carefully.
Old ruins. Choke points. Elevated sightlines.
But also… memory.
It was where Halcyon had begun its field trials.
Where Seren had taken her first confirmed shot.
Where everything that mattered was born—and where, if fate allowed it, everything would end.
“She’ll come,” Moth said, voice low.
“And if she doesn’t?” Lena asked.
Moth stepped into the light. The years had aged him differently than most. His eyes had grown colder, but his pulse had not slowed.
“She will. Because she’s not chasing me.”
He looked at his own reflection in the blackened monitor.
“She’s chasing who I used to be.”

Chapter 7: The Weight of a Single Shot
Ash moved like water through the ruins.
Each step was a negotiation with time—quiet, patient, exact. The old Halcyon training grounds had long since been reclaimed by moss and silence. Cracked stone courtyards where gunfire once echoed were now blanketed in fog. Firing ranges turned to graveyards. Buildings collapsed inward, like memories trying to forget themselves.
But she remembered.
Every wall.
Every window.
Every line of fire.
Ash crouched beneath what had once been the overwatch tower—her old nest. The wind hadn’t changed. Not yet. But the stillness had. There was a presence in the air, like something watching from behind the veil.
Moth was near.
She didn’t need proof. She felt it—like gravity pulling at her ribs.
She set down her pack. Unclipped Valkyrie. Began the ritual.
The rifle came apart not out of need, but memory. She checked every part. Ran her fingers along the bolt. Wiped the optics. Whispered to the scope like it might respond. This rifle had taken lives, yes—but more than that, it had witnessed.
Each target. Each moment. Each choice she could never take back.
She laid out her final three bullets.
One was marked with a red line.
His name.
If it came to it, she would use that one last.
Ash checked her watch.
3 hours until first light.
Moth liked the dawn. He had once said that a sniper’s best ally was contradiction—kill in the hour people felt safest. Shoot just as the birds sing. Make the world doubt itself.
She had doubted herself enough for a lifetime.
Ash reached into her pack and pulled out the old photo.
The one she shouldn’t have kept.
Alexei. Victor. Herself.
They were so young. She had kept her hair short even then—too practical for vanity. Victor had that damned smirk—the one that said he knew five things you didn’t, and one of them could save your life.
She hadn’t smiled in that photo. But she remembered the moment it was taken.
They had just completed the “Final Calibration” exercise—snipers paired off, given conflicting orders, forced to choose whether to complete the mission or trust their partner’s instinct. It was a test of loyalty. And morality. And command.
She and Victor had both disobeyed.
They had chosen each other.
That night, Alexei bought them vodka and said nothing.
The silence was his approval.
Ash tucked the photo away. Looked to the sky.
Still no stars.
Too much cloud cover. That was good. She preferred cover to clarity.
In the distance, a soft click echoed.
Too faint for most.
Not for her.
She froze. Heart slowed. Focus sharpened.
A footfall. A weapon being braced. Not close—maybe 300 meters. Someone setting up, not attacking. Moth, or one of his ghosts?
It didn’t matter.
She pulled Valkyrie back together, smooth as breath. Chambered the red-marked round. Clicked the safety off. Slid prone across the cold concrete, her eyes behind glass, her breath steady.
And there he was.
Through the scope. Not a shadow. Not a decoy.
Him.
Older. Leaner. But still Victor in all the ways that haunted.
Standing atop the ruins of what was once Alexei’s office, framed by a shattered skylight. Looking right at her. No scope. No weapon in hand.
Just watching.
Ash didn’t pull the trigger.
Not yet.
Because he raised a hand.
Not in surrender.
But in recognition.
The same way he used to signal: Trust me. Not yet.
The wind shifted.
And in that moment, Ash realized—
This wasn't just the endgame.
It was the beginning of the truth.

Chapter 8: Cathedrals of Concrete and Ghosts
The ruins were sacred in their own way.
Not to any god, but to a discipline—angles, range, line of sight. Every crack in the wall was a margin of error; every hallway a tunnel of fate. Ash moved through it like a spirit retracing old steps, her body a map of training scars and muscle memory.
Moth had vanished.
The rooftop where he had stood moments ago was now empty—no shell casings, no sensor traces, no heat signatures. Just absence. Intentional. A message.
This isn't your kill yet.
Ash lowered Valkyrie. Her hands were steady. Her heart was not.
She moved deeper into the complex, weaving through collapsed corridors and shattered frames. Her mind stayed in the moment, but her memory kept flickering—images surfacing like bubbles from deep water.
She remembered learning to shoot here.
Alexei standing behind her, pressing a hand to her shoulder. “You don’t pull the trigger,” he’d said. “You release it. Like letting go of something you love.”
She remembered Victor adjusting her aim, teasing her under breath. “You’re aiming like a tactician, Seren. Try aiming like someone who’s already forgiven herself.”
That was the cruelest part.
He had never hated her. Not even in the end.
And yet, he betrayed her all the same.
Ash froze at a stairwell—half-gutted, sagging inward. She scanned the angles. No movement. But something felt wrong.
Then came the voice.
Not loud. Not amplified. Just there. Carried on the still air like an old whisper.
“You still lead with your left foot. I told you it echoes.”
Ash turned, fast. Valkyrie up.
Nothing.
No heat signature. No silhouette. Just static bleeding into her earpiece.
“You shouldn’t be here, Seren.”
Her name. Her real name. No one had spoken it in years.
She kept moving. Slow. Rifle tracking every corridor like a pendulum of fate.
“You think you’re here to kill me. But you're not. Not really.”
The voice wasn’t taunting. It wasn’t even confident.
It was tired.
Ash reached the center chamber—the old operations control room. Roof half gone. Consoles gutted. Moss blooming in the corners. She swept the space. Nothing.
Then: a flutter.
A butterfly charm.
Dangling from a bent aerial, spinning slowly.
Below it, a satchel.
She moved forward, wary. No wires. No traps. Just… a gift.
Inside: an old field recorder. Still warm.
She played it.
“There were three of us. You remember. You, me, Alexei. The last time we stood here, we were gods. At least, we thought we were.”
“I betrayed you. Yes. But I didn’t kill you. That was your choice. You killed Seren. I only helped you become Ash.”
Ash stood frozen. Her jaw clenched. Her grip tightened.
“And now you're here to erase the last piece of your past.”
“But if you do, you'll never know why.”
The recording clicked off.
No coordinates. No riddles.
Just that word—why—dangling like the charm above it.
Ash exhaled. Not because she was calm, but because she realized something:
She wasn’t the one hunting anymore.
Not entirely.
This was his labyrinth now.
But she knew how to break walls.
She lifted Valkyrie.
And whispered to the ghost in her earpiece, “Run all you want. But I still remember how you breathe.”
Then she stepped into the lightless corridor ahead—toward the heart of the ruins, where the final truth waited.
And perhaps, the shot that would end both of them.

Chapter 9: The Distance Between Crosshairs
Ash moved deeper into the ruins.
Each step echoed not in the air, but in her spine. Her body remembered this place—where her hands first learned to steady a rifle, where her trust had once felt indestructible, where everything began to fracture.
The corridor narrowed, leading into the inner sanctum of the old Halcyon complex. Once, this room had held servers, surveillance boards, orders never written. Now it was hollow. Except for the two of them.
Moth stood at the far end.
Not hiding. Not aiming.
Waiting.
Ash entered silently, Valkyrie half-raised, not pointed.
He didn't flinch.
The space between them was thirty meters. A sniper’s insult of a distance—too close for comfort, too far for trust. And yet neither moved.
For a long moment, the only sound was the wind threading through the broken ceiling above them.
Ash broke it first.
"Why here?"
Moth didn’t smile. For once.
“Because this is where we stopped being real.”
Ash's finger rested on the trigger guard. Not threatening. Prepared.
“You could’ve killed me ten times over,” she said.
“I still could.”
“Then why didn’t you?”
He looked away. Not in fear, but in thought.
“Because you’re the only person who might understand.”
Ash’s eyes narrowed. The scar under her left brow twitched—the one from the shrapnel Victor once pulled out, years ago, with shaking hands and teeth clenched against the pain.
“Understand what?”
Moth stepped forward, just one pace.
“What Halcyon really was. What Alexei knew. Why he died.”
Silence cracked like glass.
Ash stepped forward too, mirroring him.
“He died in an ambush.”
“That’s the story they sold you.”
He reached into his coat—slowly, deliberately—and pulled out a sealed folder. Tossed it across the floor. It skidded to her boots.
“What’s this?”
“Alexei’s last debrief. He left it for you. But they intercepted it. I stole it back. Took me two years.”
Ash didn’t pick it up. Not yet.
“Why give it to me now?”
Moth's voice dropped.
“Because you need to know what we were built for. Not the wars we fought, but the lies we protected.”
Ash finally lowered Valkyrie.
Just slightly.
“And what were you built for?” she asked, her voice suddenly colder.
Moth looked her in the eye.
And for the first time, Victor looked back.
“To disappear. So people like you could stay whole.”
They stood in silence again—two ghosts in a cathedral of ruin. The distance between them measured in choices, not meters.
Ash knelt. Picked up the folder.
Hands steady.
She opened it.
Her breath caught.
Photographs. Codenames. Black sites she had never been told about. Orders issued in Alexei’s name after his supposed death. Surveillance footage of her—before she was recruited. Her family.
Her real file.
She looked up, eyes burning.
“Why didn’t you tell me?”
“Because you weren’t ready.”
“And now I am?”
Moth nodded once.
“Because if you pull that trigger now, you’ll know exactly what you’re killing.”
Ash looked at Valkyrie.
Then at him.
Then… she turned and walked away.
Not out of mercy.
Not out of fear.
But because the bullet no longer belonged to her.
Not yet.

Prologue: The Breath Before the Trigger
The wind was quiet that morning—too quiet for the season.
A pale sun crawled over the jagged horizon, casting long shadows across the frostbitten valley. The mountains stood like forgotten gods, their shoulders dusted in snow, silent watchers over the world below. Between them lay a sliver of steel and wood—perfectly still, like it had grown from the earth. It was not just a rifle. It was history, vengeance, mathematics, and instinct. It was precision made manifest.
They called it Valkyrie.
Not for its elegance, though it had that. Nor for its legacy, though bloodied scrolls of war would tell tales of its lineage. No, it earned its name for what it summoned: the judgment of the unseen. The last breath. The final beat before silence. For in the right hands, it did not miss.
Long before it came into the possession of its current wielder, Valkyrie had traveled far—from secret factories hidden beneath tundras, through ghost wars fought in deserts no map dared mark, to auction tables in shadowed rooms where generals and ghosts bartered with more than money. Each owner left a part of themselves in the cold grooves of its barrel. Each trigger pull etched their memory into its metal. And still it endured, patient.
Its latest bearer—nameless for now—lay prone on the ridge, the earth damp beneath their chest, their breath slow, matching the rhythm of the grass swaying beside them. They were not new to the craft. They did not worship the weapon. They understood it. The way a poet understands silence. The way a surgeon respects the scalpel.
Below, far below, a convoy moved through the ravine like ants unaware of the boot above. Trucks, men, secrets. There were targets there. Perhaps one. Perhaps many. Perhaps none. The data was uncertain. The order was not.
"Wait for the wind."
They remembered that lesson well. The first rule, passed down from a teacher long vanished: patience is the sniper’s most dangerous caliber. Not the bullet. Not the scope. But the breath before the trigger. That fraction of a moment when time pauses, when fate balances on the edge of a hairline.
And so, they waited.
As the light grew, so did the stillness. Somewhere far away, a raven screamed into the silence, and the sound fell flat against the sky. The rifle remained still. The eye behind the scope watched. Not for movement. But for meaning.
This is not the story of a single shot. Nor is it a tale of war and glory. This is the story of the sniper rifle itself—of those who carried it, those who feared it, and those who never even heard the crack before the world went dark.
This is the story of Valkyrie.
And this is only the beginning.

Chapter 2: Echoes in Brass
Empty shell casings have memories.
Long after the echo fades, after the smoke has drifted into myth, they remain—burnt, hollow, discarded. But they remember. The trembling finger. The exhale. The choice. Some call it the price of duty. Others, the weight of killing. But for those who live by the scope, it is neither burden nor pride. It is necessity.
The shell from this morning’s shot now lay buried in the grass, glinting dully in the half-light. One shot. One hit. No confirmation needed. Valkyrie did not lie. And yet, the figure who pulled the trigger did not move. The mission was not over. It had just begun.
The sniper—codename Ash—stayed low. Always low.
The name was not theirs by birth, but by fire. Years ago, in the hills of a nation no longer on any map, a village had burned because Ash had not pulled the trigger in time. A convoy rolled through. A deal was struck. An atrocity was made. By the time command issued the go-order, the screams had already started.
Ash survived the inquiry. The commanding officer did not. But the name remained. As a reminder. As penance. As armor.
Now, Ash moved silently through the underbrush, every step a practiced prayer. The target—known only as Moth—had not been among the fallen. Intelligence had lied. Or worse, had been paid to lie. Ash suspected both.
The rifle slung across their back groaned slightly, metal against metal. Valkyrie did not like the quiet after a kill. It was a weapon born to hunt. And Moth was prey worth hunting.
In the years since their paths first crossed, Moth had grown into a phantom. No face. No confirmed voice. Only signals, shadows, and aftermaths. Bombings without cause. Operations that vanished from record. Bodies too mutilated for identification. Always followed by the same pattern: a butterfly charm, scorched and melted, left like a calling card.
Ash had once found one lodged in a child’s skull.
That was when the vendetta began.
A mission briefing had never been so personal. This wasn't about politics. Or even justice. This was about understanding why. Why a man—or whatever Moth truly was—would kill to be seen, yet never step into the light. Why a sniper’s bullet never found him. Why he always seemed to know… just a moment before.
Some said Moth could hear the shot before it was fired.
Ash didn’t believe in ghosts.
But they believed in patterns. And there was one emerging now.
The convoy had not carried weapons. It had carried data. Hardened drives. Radiation-shielded cores. Surveillance-grade encryption. Whatever was on them had enough value to warrant decoys, mercenaries, and, most tellingly—sacrifice.
Ash’s scope had captured one frame, just before the impact. A face. Not Moth’s. But someone looking up. Not afraid. Aware.
The kind of look someone gives a camera they didn’t expect to find.
Ash had already pulled the image, burned it into a drive, and left it for dead-drop extraction. The trail was warm. The time to hunt had come.
Valkyrie was cleaned that night. Piece by piece. Ritual by ritual. Each part laid out on cloth like sacred scripture. The action was silent, practiced. Every sniper has their rites, and Ash had theirs: dismantle the weapon to remember the weight of its unity. Oil the bolt to honor the precision. Align the scope to never forget what a single blink can cost.
Then came the final motion: placing the next round into the chamber.
Ash didn’t use full magazines. One bullet. Always one.
Because when the moment comes, there is only ever one shot.
Tomorrow, the chase would resume. But tonight, in the rusted ruins of an abandoned outpost, beneath a moon that refused to be full, Ash whispered to the rifle like a lover:
"We’re not done yet."
And somewhere, hundreds of miles away, a man who went by Moth paused mid-sentence… and smiled.

Chapter 3: The Art of Vanishing
In a nameless city built on secrets and salt, beneath a club that pulsed with manufactured hedonism, Moth lit a cigarette with fingers that had never trembled.
He had been watching the monitors for hours—sixteen screens arranged in perfect angles, each one streaming feeds from the convoy that had failed to reach its destination. Not one showed the face he expected.
Ash.
The sniper had been there. The report was late by forty-three seconds. That was enough. Moth had built empires on smaller discrepancies. The kill had been clean—center mass, wind-compensated, no trace of panic. Ash was improving. Or perhaps evolving.
Moth exhaled slowly, letting the smoke curl toward the ceiling where the air purifiers would devour it like obedient predators. He never smoked in public. Only here, in this sealed room, beneath layers of denial and encryption, did he allow himself that indulgence.
His reflection shimmered faintly in the glass. A face built from memory, not birth. Bone structure altered. Retina replaced. Voice tuned to frequencies that bypassed most lie detectors. There was nothing original left. Moth had murdered his former self long ago.
He hadn't always been a monster.
Once, long before the name Moth clung to him like soot, he had believed in things. Borders. Orders. Right and wrong. The sniper rifle that Ash now carried had once belonged to a man Moth had trained beside—the man, in fact, who had taught them both.
But where Ash took the teachings and buried them like roots, Moth burned them down to ash.
It was ideology that betrayed him. He remembered the tribunal. The fabricated charges. The smirk of the man who pulled the strings. How many nations had he bled for, only to be cast out when the blood dried?
He became invisible not by choice, but by necessity.
Yet invisibility became power.
He learned to build networks of silence. Hired ghosts. Operated in code so subtle it passed for coincidence. He discovered that nothing in this world was more feared than a face without a name.
Ash, though, was the exception.
Ash remembered his name.
That was the problem.
Moth flicked the cigarette into the glass tray and turned toward a display console where a single red dot pulsed near the border of what used to be a war zone. The signal was faint, but consistent.
"She's close," he muttered.
A woman entered the room without knocking—short hair, tactical posture, a pistol holstered upside-down beneath her jacket. Lena. Former SIS, current operations chief. Efficient, ruthless, loyal only to the contract.
“She’s already moved beyond the ridge. We have ten hours at most before she hits the secondary cache.”
Moth tapped the screen, zooming into the signal.
“She doesn’t know what’s in it.”
“She doesn’t need to,” Lena replied. “Only that you want it.”
Moth smiled, cold and deliberate.
“She’ll take the bait.”
“You're sure?” Lena asked. “Ash isn’t like the others.”
“No,” Moth said. “She’s worse.”
He walked to the locker, input a code that changed every hour, and retrieved a weapon unlike any he’d used in years. Short-barreled, suppressed, matte black. Not a sniper’s rifle. A killer’s whisper.
“She’s coming for me,” he said. “Let her.”
Then he looked at the wall, where an old photo—faded, grainy, perhaps deliberately damaged—showed three people in uniform. Two men, one woman. Young. Smiling. One of them held a rifle marked with an old inscription in Cyrillic.
Valkyrie.
Moth tapped the photo once, gently.
“You trained her well, old friend,” he whispered. “Let’s see if she learned everything.”

Chapter 4: No One Hears the Grass Grow
Ash had always preferred the silence before dawn.
It was not the silence of absence but of possibility. Before the first bird dared to sing, before the machines began their endless hum, the world paused—just long enough for a hunter to listen. And right now, that silence spoke volumes.
Ash crouched beside a stream no wider than a boot. The water was slow, cautious—like it, too, sensed something unnatural in the air. Across the clearing, beneath a bluff wrapped in brittle moss and shattered roots, lay the cache. Not buried. Not hidden. Just placed—as if someone wanted it found.
A trap.
Ash’s breath moved like smoke through cloth. The ghillie suit fluttered faintly, blending with the reeds. Every inch of their movement was calculated: heel to toe, muscle to moss. The rifle, cradled like an extension of the spine, remained still.
Valkyrie hadn’t spoken yet.
Ash always imagined the rifle had moods—not voices, not commands, but instincts. When something was wrong, the scope would fog slightly, or the bolt would catch half a breath longer than expected. This morning, Valkyrie was nervous.
Still, Ash moved.
It took 47 minutes to cross 600 meters. Every approach path had already been plotted the night before. Every exit route, memorized. There was no backup. No drone. No failsafe.
Ash trusted no one.
A footfall away from the cache, something shifted.
Not in the landscape. In the pattern.
A thread pulled loose in the air.
Ash’s eyes locked onto a glint—imperceptible to most, but to a sniper’s trained gaze, it screamed. A fiber-optic filament, broken at the tip, likely feeding light to a passive sensor. Not amateur work. Industrial-grade camouflage.
Moth was watching.
Ash didn’t flinch. Didn’t blink. Instead, in one motion so fluid it felt rehearsed, they dropped flat, rolled sideways, and aimed upward—not at the cache, but at the outcropping above and behind it.
There.
Just for an instant: movement.
A flicker of fabric. The shimmer of optics.
Ash didn’t shoot.
Not yet.
Instead, they waited. A second. Two. A shadow crossed the bluff, but it didn’t run. It circled. Professional. Not Moth—he didn’t do his own dirty work anymore. But one of his new ghosts, perhaps. Ash would test them.
A distraction grenade rolled from the side pack. Silent fuse. No light, no sound—just a pressure wave designed to rupture equilibrium. Ash launched it toward the base of the bluff, then moved—fast, low, toward the stream, away from the cache.
The detonation was muted. A bird shrieked. No gunfire.
Ash counted: One… Two… Three…
And then the rifle barked.
Not Valkyrie. Something smaller. Suppressed.
The shot hit the tree behind them—wrong angle, off mark.
Amateur mistake.
Ash pivoted, dropped into firing stance, and Valkyrie sang.
A single shot. No echo. No hesitation.
The bluff returned to silence.
Ash waited for movement. None came. Then rose and moved—swift and ghostlike—to the source. There, amid torn ferns and blood-slicked stone, lay a body in tactical black. No insignia. No dog tag. Just a mark on the glove: a stylized moth etched in silver.
Ash knelt, turned the head gently. Young. Probably mercenary. Indoctrinated, not trained. There would be more.
The cache was never the goal.
It was the invitation.
Ash looked skyward. The sun was barely rising, but it already felt late.
Moth was drawing closer. No more intermediaries. No more games. The field was narrowing.
Ash whispered, “I’m coming.”
Valkyrie remained silent. But it felt like approval.

Chapter 5: Before the Names
Five years ago, the world still made sense—at least on paper.
There was no Ash then. No Moth. Just two operatives in the service of a program that officially didn’t exist: Project Halcyon, a deniable initiative buried within the intelligence branches of six nations, designed to train and deploy off-the-record snipers to places where flags dared not fly.
They called themselves The Silent Three.
Alexei Morozov, the mentor—ex-Spetsnaz, with eyes that saw through lies like glass. The one who taught them how to slow their heartbeat, how to disappear behind the wind, how to kill without hating.
Second, there was her—young, precise, calculating. Back then, her name was Seren Ward. A former military analyst with perfect visual recall and a surgeon’s patience. They said she could hit a moving target at 1,000 meters and forget she ever pulled the trigger.
And the third…
He was charming then. Always one step ahead. They knew him as Victor Delane. The instructors called him the ghost that smiles. He had a gift—not just for shooting, but for slipping into roles, into accents, into people’s trust. He could kill and make you thank him for it.
Seren and Victor were a perfect match.
Not romantically—though there had been moments, late at night, whispers that never turned into kisses. But tactically. Intellectually. They finished each other’s calibrations. Anticipated each other’s firing positions without speaking. They moved like mirrored thoughts.
Alexei pushed them hard, but not cruelly. He saw something rare: two souls shaped by war, but unbroken by it. He told them once, over tea and silence, that one day the wind would change—and that when it did, they would either hold each other up or destroy each other completely.
He was right.
It happened in the desert.
A black op so sensitive it never made the logs. A warlord in possession of a data payload—encrypted files that hinted at corruption not just in foreign regimes, but in their own chain of command. Halcyon was ordered to retrieve the payload. No witnesses. No survivors.
Seren hesitated.
She saw the faces of children in the encampment. They weren’t just collateral. They were the bargain. The price for silence. She broke protocol. Warned the locals. Extracted the data, but left the warlord alive. She thought Victor had her back.
She was wrong.
Victor filed the report. Seren was labeled a liability.
Alexei disappeared a week later—officially “killed in an ambush,” but Seren had always suspected something else. Victor… simply vanished. Not discharged. Not reprimanded. Transferred—to where, no one said.
It wasn’t until a year later, during a recon mission in Eastern Europe, that she saw the calling card for the first time:
A butterfly charm.
Charred at the edges.
She didn’t need a name to know who had left it.
Victor had become Moth.
And Seren shed her own name like dead skin.
From then on, she was only Ash.
A new ghost for a new war.
Ash woke from the memory with a start.
The campfire had gone cold. Morning mist curled around the rocks like smoke from a distant fire. In her hands, Valkyrie felt heavy—not with weight, but with memory.
The chase was no longer professional. It hadn’t been for years.
It was personal now.
She didn’t hunt Moth because of orders. She hunted him because she had loved who he used to be—and mourned who he became.
And she wasn’t sure which grief was heavier.
Ash stood, chambered a round, and looked east.
The wind was changing.

Chapter 6: A Smile in the Scope
He hadn’t dreamed of her in years.
But last night, she returned—not as Ash, not as an adversary cloaked in shadows and vengeance, but as Seren. In the dream, they stood on the edge of a ruined watchtower, wind howling around them, maps clutched in their hands. Her eyes were sharp, her expression unreadable. She said nothing. Just looked at him. And he knew what the silence meant.
“You left before I asked you to.”
Moth awoke to the bitter taste of metal. Not blood—regret. It was always sharper than anything forged.
He sat up in the armory chamber of one of his satellite compounds—a monolithic shelter buried beneath a forgotten railway station in the Baltics. Concrete walls bled with condensation, and rows of weaponry stared back at him like old accusations. Everything here had a story. And every story pointed to one conclusion:
Ash was close.
Lena stood by the console when he emerged. She didn’t flinch at his sudden presence.
“She neutralized the bluff team,” she said without turning.
“Expected,” Moth replied. “They were just the opening act.”
“Then why send them?”
“To remind her this is still a performance.”
He walked past her and tapped into the comms terminal. A real-time satellite feed spun into motion—infrared sweeps, radio packet sniffers, deep acoustic surveillance grids. He traced Ash’s path by the absence of disruption. Where others stumbled, she moved like a shadow between atoms.
“She’ll go for the cache in Sector 9 next,” Lena offered.
“She’ll skip it,” Moth corrected. “She’ll assume it’s a diversion.”
“It is a diversion.”
“Exactly.”
Lena finally turned. “You want her to come here.”
Moth smiled—though it barely reached his eyes.
“I need her to.”
He paused at the old locker—the one he hadn’t opened in years. Not since Halcyon fell.
Inside, preserved in foam and memory, was his first sniper rifle.
It was nothing like Valkyrie. Less elegant. Less forgiving. But it had history. He’d once let Seren hold it. She had called it ugly, but she smiled when she said it. That was rare for her. Back then, her smiles were currency—precious, valuable, withheld until earned.
Moth picked up the rifle and held it to his shoulder.
The weight was wrong.
Or maybe he was.
He set it down and opened the side compartment instead, retrieving a small, thin box. Inside lay a photo, untouched by time—faded only at the corners. Three people. A field. Laughter caught mid-breath.
Alexei. Seren. Himself.
None of them had names then.
Just codes. Just belief.
“I didn’t leave you,” he said quietly, to no one. “I saved you.”
Lena said nothing. She had heard these half-confessions before.
Moth closed the box and turned to face the map on the far wall—an embossed topographical relief of the coming battlefield.
The location had been chosen carefully.
Old ruins. Choke points. Elevated sightlines.
But also… memory.
It was where Halcyon had begun its field trials.
Where Seren had taken her first confirmed shot.
Where everything that mattered was born—and where, if fate allowed it, everything would end.
“She’ll come,” Moth said, voice low.
“And if she doesn’t?” Lena asked.
Moth stepped into the light. The years had aged him differently than most. His eyes had grown colder, but his pulse had not slowed.
“She will. Because she’s not chasing me.”
He looked at his own reflection in the blackened monitor.
“She’s chasing who I used to be.”

Chapter 7: The Weight of a Single Shot
Ash moved like water through the ruins.
Each step was a negotiation with time—quiet, patient, exact. The old Halcyon training grounds had long since been reclaimed by moss and silence. Cracked stone courtyards where gunfire once echoed were now blanketed in fog. Firing ranges turned to graveyards. Buildings collapsed inward, like memories trying to forget themselves.
But she remembered.
Every wall.
Every window.
Every line of fire.
Ash crouched beneath what had once been the overwatch tower—her old nest. The wind hadn’t changed. Not yet. But the stillness had. There was a presence in the air, like something watching from behind the veil.
Moth was near.
She didn’t need proof. She felt it—like gravity pulling at her ribs.
She set down her pack. Unclipped Valkyrie. Began the ritual.
The rifle came apart not out of need, but memory. She checked every part. Ran her fingers along the bolt. Wiped the optics. Whispered to the scope like it might respond. This rifle had taken lives, yes—but more than that, it had witnessed.
Each target. Each moment. Each choice she could never take back.
She laid out her final three bullets.
One was marked with a red line.
His name.
If it came to it, she would use that one last.
Ash checked her watch.
3 hours until first light.
Moth liked the dawn. He had once said that a sniper’s best ally was contradiction—kill in the hour people felt safest. Shoot just as the birds sing. Make the world doubt itself.
She had doubted herself enough for a lifetime.
Ash reached into her pack and pulled out the old photo.
The one she shouldn’t have kept.
Alexei. Victor. Herself.
They were so young. She had kept her hair short even then—too practical for vanity. Victor had that damned smirk—the one that said he knew five things you didn’t, and one of them could save your life.
She hadn’t smiled in that photo. But she remembered the moment it was taken.
They had just completed the “Final Calibration” exercise—snipers paired off, given conflicting orders, forced to choose whether to complete the mission or trust their partner’s instinct. It was a test of loyalty. And morality. And command.
She and Victor had both disobeyed.
They had chosen each other.
That night, Alexei bought them vodka and said nothing.
The silence was his approval.
Ash tucked the photo away. Looked to the sky.
Still no stars.
Too much cloud cover. That was good. She preferred cover to clarity.
In the distance, a soft click echoed.
Too faint for most.
Not for her.
She froze. Heart slowed. Focus sharpened.
A footfall. A weapon being braced. Not close—maybe 300 meters. Someone setting up, not attacking. Moth, or one of his ghosts?
It didn’t matter.
She pulled Valkyrie back together, smooth as breath. Chambered the red-marked round. Clicked the safety off. Slid prone across the cold concrete, her eyes behind glass, her breath steady.
And there he was.
Through the scope. Not a shadow. Not a decoy.
Him.
Older. Leaner. But still Victor in all the ways that haunted.
Standing atop the ruins of what was once Alexei’s office, framed by a shattered skylight. Looking right at her. No scope. No weapon in hand.
Just watching.
Ash didn’t pull the trigger.
Not yet.
Because he raised a hand.
Not in surrender.
But in recognition.
The same way he used to signal: Trust me. Not yet.
The wind shifted.
And in that moment, Ash realized—
This wasn't just the endgame.
It was the beginning of the truth.

Chapter 8: Cathedrals of Concrete and Ghosts
The ruins were sacred in their own way.
Not to any god, but to a discipline—angles, range, line of sight. Every crack in the wall was a margin of error; every hallway a tunnel of fate. Ash moved through it like a spirit retracing old steps, her body a map of training scars and muscle memory.
Moth had vanished.
The rooftop where he had stood moments ago was now empty—no shell casings, no sensor traces, no heat signatures. Just absence. Intentional. A message.
This isn't your kill yet.
Ash lowered Valkyrie. Her hands were steady. Her heart was not.
She moved deeper into the complex, weaving through collapsed corridors and shattered frames. Her mind stayed in the moment, but her memory kept flickering—images surfacing like bubbles from deep water.
She remembered learning to shoot here.
Alexei standing behind her, pressing a hand to her shoulder. “You don’t pull the trigger,” he’d said. “You release it. Like letting go of something you love.”
She remembered Victor adjusting her aim, teasing her under breath. “You’re aiming like a tactician, Seren. Try aiming like someone who’s already forgiven herself.”
That was the cruelest part.
He had never hated her. Not even in the end.
And yet, he betrayed her all the same.
Ash froze at a stairwell—half-gutted, sagging inward. She scanned the angles. No movement. But something felt wrong.
Then came the voice.
Not loud. Not amplified. Just there. Carried on the still air like an old whisper.
“You still lead with your left foot. I told you it echoes.”
Ash turned, fast. Valkyrie up.
Nothing.
No heat signature. No silhouette. Just static bleeding into her earpiece.
“You shouldn’t be here, Seren.”
Her name. Her real name. No one had spoken it in years.
She kept moving. Slow. Rifle tracking every corridor like a pendulum of fate.
“You think you’re here to kill me. But you're not. Not really.”
The voice wasn’t taunting. It wasn’t even confident.
It was tired.
Ash reached the center chamber—the old operations control room. Roof half gone. Consoles gutted. Moss blooming in the corners. She swept the space. Nothing.
Then: a flutter.
A butterfly charm.
Dangling from a bent aerial, spinning slowly.
Below it, a satchel.
She moved forward, wary. No wires. No traps. Just… a gift.
Inside: an old field recorder. Still warm.
She played it.
“There were three of us. You remember. You, me, Alexei. The last time we stood here, we were gods. At least, we thought we were.”
“I betrayed you. Yes. But I didn’t kill you. That was your choice. You killed Seren. I only helped you become Ash.”
Ash stood frozen. Her jaw clenched. Her grip tightened.
“And now you're here to erase the last piece of your past.”
“But if you do, you'll never know why.”
The recording clicked off.
No coordinates. No riddles.
Just that word—why—dangling like the charm above it.
Ash exhaled. Not because she was calm, but because she realized something:
She wasn’t the one hunting anymore.
Not entirely.
This was his labyrinth now.
But she knew how to break walls.
She lifted Valkyrie.
And whispered to the ghost in her earpiece, “Run all you want. But I still remember how you breathe.”
Then she stepped into the lightless corridor ahead—toward the heart of the ruins, where the final truth waited.
And perhaps, the shot that would end both of them.

Chapter 9: The Distance Between Crosshairs
Ash moved deeper into the ruins.
Each step echoed not in the air, but in her spine. Her body remembered this place—where her hands first learned to steady a rifle, where her trust had once felt indestructible, where everything began to fracture.
The corridor narrowed, leading into the inner sanctum of the old Halcyon complex. Once, this room had held servers, surveillance boards, orders never written. Now it was hollow. Except for the two of them.
Moth stood at the far end.
Not hiding. Not aiming.
Waiting.
Ash entered silently, Valkyrie half-raised, not pointed.
He didn't flinch.
The space between them was thirty meters. A sniper’s insult of a distance—too close for comfort, too far for trust. And yet neither moved.
For a long moment, the only sound was the wind threading through the broken ceiling above them.
Ash broke it first.
"Why here?"
Moth didn’t smile. For once.
“Because this is where we stopped being real.”
Ash's finger rested on the trigger guard. Not threatening. Prepared.
“You could’ve killed me ten times over,” she said.
“I still could.”
“Then why didn’t you?”
He looked away. Not in fear, but in thought.
“Because you’re the only person who might understand.”
Ash’s eyes narrowed. The scar under her left brow twitched—the one from the shrapnel Victor once pulled out, years ago, with shaking hands and teeth clenched against the pain.
“Understand what?”
Moth stepped forward, just one pace.
“What Halcyon really was. What Alexei knew. Why he died.”
Silence cracked like glass.
Ash stepped forward too, mirroring him.
“He died in an ambush.”
“That’s the story they sold you.”
He reached into his coat—slowly, deliberately—and pulled out a sealed folder. Tossed it across the floor. It skidded to her boots.
“What’s this?”
“Alexei’s last debrief. He left it for you. But they intercepted it. I stole it back. Took me two years.”
Ash didn’t pick it up. Not yet.
“Why give it to me now?”
Moth's voice dropped.
“Because you need to know what we were built for. Not the wars we fought, but the lies we protected.”
Ash finally lowered Valkyrie.
Just slightly.
“And what were you built for?” she asked, her voice suddenly colder.
Moth looked her in the eye.
And for the first time, Victor looked back.
“To disappear. So people like you could stay whole.”
They stood in silence again—two ghosts in a cathedral of ruin. The distance between them measured in choices, not meters.
Ash knelt. Picked up the folder.
Hands steady.
She opened it.
Her breath caught.
Photographs. Codenames. Black sites she had never been told about. Orders issued in Alexei’s name after his supposed death. Surveillance footage of her—before she was recruited. Her family.
Her real file.
She looked up, eyes burning.
“Why didn’t you tell me?”
“Because you weren’t ready.”
“And now I am?”
Moth nodded once.
“Because if you pull that trigger now, you’ll know exactly what you’re killing.”
Ash looked at Valkyrie.
Then at him.
Then… she turned and walked away.
Not out of mercy.
Not out of fear.
But because the bullet no longer belonged to her.
Not yet.

"#,
        ),
    )];

    let bpe = o200k_base().unwrap();
    let token_count = bpe
        .encode_with_special_tokens(messages[0].content.as_str())
        .len();
    tracing::info!("Token count: {}", token_count);

    openai
        .model_id(String::from("gpt-4o-mini"))
        .messages(messages)
        .temperature(1.0);

    let response = openai.chat().await.unwrap();
    println!("{:#?}", response);
}
