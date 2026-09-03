//! OpenAI-compatible chat completion request and response types.

use atoma_core::request::{SamplingParams, Usage as EngineUsage};
use atoma_core::types::TokenCount;
use std::collections::HashMap;
use thiserror::Error;
use utoipa::ToSchema;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::ser::{CompactFormatter, Formatter};
use serde_json::Value;

// TODO: fields that are named `r#type` should have values that represent
// actual expected types that are deserializable from a string instead of
// just `String` since a user could input anything if we allow them to.
// On our end, it's also beneficial since we will want to match on that
// type. For now a naive version of this is OK, but may want to do this
// before stabilizing the API to avoid misuse.

/// ID of the model to use.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "model", deserialize = "model"))]
pub enum Model {
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-2-7b",
        deserialize = "meta-llama/Meta-Llama-2-7b"
    ))]
    Llama27b,
    #[serde(rename(
        serialize = "meta-llama/Llama-2-7b-chat-hf",
        deserialize = "meta-llama/Llama-2-7b-chat-hf"
    ))]
    Llama27bChatHf,
    #[serde(rename(
        serialize = "meta-llama/Llama-2-70b-hf",
        deserialize = "meta-llama/Llama-2-70b-hf"
    ))]
    Llama270b,
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-3-8B",
        deserialize = "meta-llama/Meta-Llama-3-8B"
    ))]
    Llama38b,
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-3-8B-Instruct",
        deserialize = "meta-llama/Meta-Llama-3-8B-Instruct"
    ))]
    Llama38bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-3-70B",
        deserialize = "meta-llama/Meta-Llama-3-70B"
    ))]
    Llama370b,
    #[serde(rename(
        serialize = "meta-llama/Meta-Llama-3-70B-Instruct",
        deserialize = "meta-llama/Meta-Llama-3-70B-Instruct"
    ))]
    Llama370bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-8B",
        deserialize = "meta-llama/Llama-3.1-8B"
    ))]
    Llama318b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-8B-Instruct",
        deserialize = "meta-llama/Llama-3.1-8B-Instruct"
    ))]
    Llama318bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-70B",
        deserialize = "meta-llama/Llama-3.1-70B"
    ))]
    Llama3170b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-70B-Instruct",
        deserialize = "meta-llama/Llama-3.1-70B-Instruct"
    ))]
    Llama3170bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-405B",
        deserialize = "meta-llama/Llama-3.1-405B"
    ))]
    Llama31405b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.1-405B-Instruct",
        deserialize = "meta-llama/Llama-3.1-405B-Instruct"
    ))]
    Llama31405bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.2-1B",
        deserialize = "meta-llama/Llama-3.2-1B"
    ))]
    Llama321b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.2-1B-Instruct",
        deserialize = "meta-llama/Llama-3.2-1B-Instruct"
    ))]
    Llama321bInstruct,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.2-3B",
        deserialize = "meta-llama/Llama-3.2-3B"
    ))]
    Llama323b,
    #[serde(rename(
        serialize = "meta-llama/Llama-3.2-3B-Instruct",
        deserialize = "meta-llama/Llama-3.2-3B-Instruct"
    ))]
    Llama323bInstruct,
    /// The ungated mirror of `meta-llama/Llama-3.1-8B-Instruct`: the same weights, reachable
    /// without a Hugging Face token, and served with the same prompt template.
    #[serde(rename(
        serialize = "NousResearch/Meta-Llama-3.1-8B-Instruct",
        deserialize = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    ))]
    NousLlama318bInstruct,
    #[serde(rename(
        serialize = "NousResearch/Hermes-3-Llama-3.1-8B",
        deserialize = "NousResearch/Hermes-3-Llama-3.1-8B"
    ))]
    HermesLlama318b,
    #[serde(rename(
        serialize = "NousResearch/Hermes-3-Llama-3.1-70B",
        deserialize = "NousResearch/Hermes-3-Llama-3.1-70B"
    ))]
    HermesLlama3170b,
    #[serde(rename(
        serialize = "NousResearch/Hermes-3-Llama-3.1-405B",
        deserialize = "NousResearch/Hermes-3-Llama-3.1-405B"
    ))]
    HermesLlama31405b,
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Llama27b => write!(f, "meta-llama/Meta-Llama-2-7b"),
            Model::Llama27bChatHf => write!(f, "meta-llama/Llama-2-7b-chat-hf"),
            Model::Llama270b => write!(f, "meta-llama/Llama-2-70b-hf"),
            Model::Llama38b => write!(f, "meta-llama/Meta-Llama-3-8B"),
            Model::Llama38bInstruct => write!(f, "meta-llama/Meta-Llama-3-8B-Instruct"),
            Model::Llama370b => write!(f, "meta-llama/Meta-Llama-3-70B"),
            Model::Llama370bInstruct => write!(f, "meta-llama/Meta-Llama-3-70B-Instruct"),
            Model::Llama318b => write!(f, "meta-llama/Llama-3.1-8B"),
            Model::Llama318bInstruct => write!(f, "meta-llama/Llama-3.1-8B-Instruct"),
            Model::Llama3170b => write!(f, "meta-llama/Llama-3.1-70B"),
            Model::Llama3170bInstruct => write!(f, "meta-llama/Llama-3.1-70B-Instruct"),
            Model::Llama31405b => write!(f, "meta-llama/Llama-3.1-405B"),
            Model::Llama31405bInstruct => write!(f, "meta-llama/Llama-3.1-405B-Instruct"),
            Model::Llama321b => write!(f, "meta-llama/Llama-3.2-1B"),
            Model::Llama321bInstruct => write!(f, "meta-llama/Llama-3.2-1B-Instruct"),
            Model::Llama323b => write!(f, "meta-llama/Llama-3.2-3B"),
            Model::Llama323bInstruct => write!(f, "meta-llama/Llama-3.2-3B-Instruct"),
            Model::NousLlama318bInstruct => {
                write!(f, "NousResearch/Meta-Llama-3.1-8B-Instruct")
            }
            Model::HermesLlama318b => write!(f, "NousResearch/Hermes-3-Llama-3.1-8B"),
            Model::HermesLlama3170b => write!(f, "NousResearch/Hermes-3-Llama-3.1-70B"),
            Model::HermesLlama31405b => write!(f, "NousResearch/Hermes-3-Llama-3.1-405B"),
        }
    }
}

impl Model {
    pub fn messages_to_prompt(&self, messages: &[Message]) -> String {
        use Model::*;
        match self {
            Llama27b | Llama27bChatHf | Llama270b => messages::messages_to_llama2_prompt(messages),
            Llama38b
            | Llama38bInstruct
            | Llama370b
            | Llama370bInstruct
            | Llama318b
            | Llama318bInstruct
            | Llama3170b
            | Llama3170bInstruct
            | Llama31405b
            | Llama31405bInstruct
            | Llama321b
            | Llama321bInstruct
            | Llama323b
            | Llama323bInstruct
            | NousLlama318bInstruct => messages::messages_to_llama3_prompt(messages),
            HermesLlama318b | HermesLlama3170b | HermesLlama31405b => {
                messages::messages_to_hermes3_prompt(messages)
            }
        }
    }
}

/// A message that is part of a conversation which is based on the role
/// of the author of the message.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// The role of the messages author, in this case system.
    System {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate
        /// between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// The role of the messages author, in this case user.
    User {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate
        /// between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// The role of the messages author, in this case assistant.
    Assistant {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// An optional name for the participant. Provides the model information to differentiate
        /// between participants of the same role.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// The refusal message by the assistant.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        /// The tool calls generated by the model, such as function calls.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    /// The role of the messages author, in this case tool.
    Tool {
        /// The contents of the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// Tool call that this message is responding to.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        tool_call_id: String,
    },
}

impl Message {
    /// Converts a message to its string representation in the prompt.
    pub fn to_prompt_string(&self) -> String {
        match self {
            Message::System { content, name: _ } => {
                let content_str = content.as_ref().map(|s| s.to_string()).unwrap_or_default();
                content_str
            }
            Message::User { content, name: _ } => {
                let content_str = content.as_ref().map(|s| s.to_string()).unwrap_or_default();
                content_str
            }
            Message::Assistant {
                content,
                name: _,
                refusal: _,
                tool_calls: _,
            } => {
                let content_str = content.as_ref().map(|s| s.to_string()).unwrap_or_default();
                content_str
            }
            Message::Tool {
                content,
                tool_call_id: _,
            } => {
                let content_str = content.as_ref().map(|s| s.to_string()).unwrap_or_default();
                content_str
            }
        }
    }
}

pub(crate) mod messages {
    use super::{Message, MessageContent, Model, ToolCall};
    use tracing::warn;

    /// Function to convert a list of messages to a prompt string in Llama2 format.
    pub(crate) fn messages_to_llama2_prompt(messages: &[Message]) -> String {
        let mut prompt = String::new();
        let mut i = 0;
        prompt.push_str("<s>");

        // Check if the first message is a system message
        if i < messages.len() && matches!(messages[i], Message::System { .. }) {
            // Start the initial [INST] block with the system prompt
            prompt.push_str("[INST] <<SYS>>\n");
            prompt.push_str(&messages[i].to_prompt_string());
            prompt.push_str("\n<</SYS>>\n\n");

            i += 1;

            // Check if the next message is a user message
            if i < messages.len() && matches!(messages[i], Message::User { .. }) {
                // Add the user's message and close the [INST] block
                prompt.push_str(&messages[i].to_prompt_string());
                prompt.push_str(" [/INST]\n");

                i += 1;
            } else {
                // No user message after system prompt, close the [INST] block
                prompt.push_str("[/INST]\n");
            }
        }

        // Process the rest of the messages
        while i < messages.len() {
            match &messages[i] {
                Message::User { .. } => {
                    // Start a new [INST] block for each user message
                    prompt.push_str("[INST] ");
                    prompt.push_str(&messages[i].to_prompt_string());
                    prompt.push_str(" [/INST]\n");
                    i += 1;

                    // Add the assistant's response if it exists
                    if i < messages.len() && matches!(messages[i], Message::Assistant { .. }) {
                        prompt.push_str(&messages[i].to_prompt_string());
                        prompt.push('\n');
                        i += 1;
                    }
                }
                Message::Assistant { .. } => {
                    // Assistant's response without preceding user message
                    prompt.push_str(&messages[i].to_prompt_string());
                    prompt.push('\n');
                    i += 1;
                }
                _ => {
                    warn!("Unsupported message type: {:?}", messages[i]);
                    i += 1;
                }
            }
        }

        prompt
    }

    /// Writes one Llama 3 role header: the role between its delimiters, then the blank line the
    /// template puts before a turn's content.
    fn push_llama3_header(prompt: &mut String, role: &str) {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(role);
        prompt.push_str("<|end_header_id|>\n\n");
    }

    /// Writes one Llama 3 turn whose content is text alone: the role header, the content if any,
    /// and the end-of-turn token.
    fn push_llama3_turn(prompt: &mut String, role: &str, content: Option<&MessageContent>) {
        push_llama3_header(prompt, role);
        if let Some(content) = content {
            prompt.push_str(&content.to_string());
        }
        prompt.push_str("<|eot_id|>");
    }

    /// Renders a conversation in the Llama 3 chat format, ending in the assistant header so that
    /// generation starts inside the assistant turn.
    pub(crate) fn messages_to_llama3_prompt(messages: &[Message]) -> String {
        let mut prompt = String::new();
        prompt.push_str("<|begin_of_text|>");

        for message in messages {
            match message {
                Message::System { content, name } => {
                    push_llama3_turn(
                        &mut prompt,
                        name.as_deref().unwrap_or("system"),
                        content.as_ref(),
                    );
                }
                Message::User { content, name } => {
                    push_llama3_turn(
                        &mut prompt,
                        name.as_deref().unwrap_or("user"),
                        content.as_ref(),
                    );
                }
                Message::Assistant {
                    content,
                    name,
                    tool_calls,
                    ..
                } => {
                    push_llama3_header(&mut prompt, name.as_deref().unwrap_or("assistant"));
                    if !tool_calls.is_empty() {
                        // Every Llama 3 model renders a tool call the same way, so one variant
                        // stands for the family.
                        let tool_calls_str = tool_calls
                            .iter()
                            .map(|tc| tc.function_call_string(Model::Llama318bInstruct))
                            .collect::<Vec<_>>()
                            .join(", ");
                        prompt.push_str("<|python_tag|>[");
                        prompt.push_str(&tool_calls_str);
                        prompt.push(']');
                    } else if let Some(content) = content {
                        prompt.push_str(&content.to_string());
                    }
                    prompt.push_str("<|eot_id|>");
                }
                Message::Tool {
                    content,
                    tool_call_id: _,
                } => push_llama3_turn(&mut prompt, "ipython", content.as_ref()),
            }
        }

        push_llama3_header(&mut prompt, "assistant");
        prompt
    }

    /// Opens one Hermes 3 turn: the start-of-turn token and the role, and nothing after them.
    fn push_hermes3_turn_start(prompt: &mut String, role: &str) {
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
    }

    /// Writes one Hermes 3 role line: the turn's opening, and the newline the template puts
    /// before a turn's content.
    fn push_hermes3_header(prompt: &mut String, role: &str) {
        push_hermes3_turn_start(prompt, role);
        prompt.push('\n');
    }

    /// The token that ends one Hermes 3 turn.
    const HERMES3_END_OF_TURN: &str = "<|im_end|>";

    /// Closes one Hermes 3 turn. The template writes the end-of-turn token straight after the
    /// turn's content, so nothing separates them.
    fn push_hermes3_end_of_turn(prompt: &mut String) {
        prompt.push_str(HERMES3_END_OF_TURN);
        prompt.push('\n');
    }

    /// Writes one Hermes 3 turn whose content is text alone: the role line, the content if any,
    /// and the end-of-turn token.
    fn push_hermes3_turn(prompt: &mut String, role: &str, content: Option<&MessageContent>) {
        push_hermes3_header(prompt, role);
        if let Some(content) = content {
            prompt.push_str(&content.to_string());
        }
        push_hermes3_end_of_turn(prompt);
    }

    /// Writes one Hermes 3 tool call as its own block: the newline the template opens a block
    /// with, the call, and the newline before the closing tag.
    fn push_hermes3_tool_call(prompt: &mut String, tool_call: &ToolCall) {
        prompt.push_str("\n<tool_call>\n");
        // Every Hermes 3 model renders a tool call the same way, so one variant stands for the
        // family.
        prompt.push_str(&tool_call.function_call_string(Model::HermesLlama318b));
        prompt.push_str("\n</tool_call>");
    }

    /// Writes one Hermes 3 assistant turn made of tool calls, one block each.
    ///
    /// The role line carries no newline of its own here: the template opens the turn with the
    /// role alone, and the first block opens with the newline that would have followed it.
    fn push_hermes3_tool_call_turn(prompt: &mut String, tool_calls: &[ToolCall]) {
        push_hermes3_turn_start(prompt, "assistant");
        for tool_call in tool_calls {
            push_hermes3_tool_call(prompt, tool_call);
        }
        push_hermes3_end_of_turn(prompt);
    }

    /// Writes one Hermes 3 tool message as a `<tool_response>` block, the way the `tool_use`
    /// template writes it. Consecutive tool messages share one turn.
    ///
    /// The role line is written only when the message before this one is not a tool message, so a
    /// conversation that opens with a tool message gets none. The end-of-turn token is written
    /// without the newline every other role's turn ends in.
    fn push_hermes3_tool_response(
        prompt: &mut String,
        content: Option<&MessageContent>,
        previous: Option<&Message>,
        next: Option<&Message>,
    ) {
        if previous.is_some_and(|previous| !matches!(previous, Message::Tool { .. })) {
            push_hermes3_header(prompt, "tool");
        }
        prompt.push_str("<tool_response>\n");
        if let Some(content) = content {
            prompt.push_str(&content.to_string());
        }
        prompt.push_str("\n</tool_response>");
        if next.is_some() {
            prompt.push('\n');
        }
        if !next.is_some_and(|next| matches!(next, Message::Tool { .. })) {
            prompt.push_str(HERMES3_END_OF_TURN);
        }
    }

    /// The system turn the Hermes 3 template writes ahead of a conversation that does not open
    /// with one of its own.
    const HERMES3_DEFAULT_SYSTEM_TURN: &str =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";

    /// Renders a conversation in the Hermes 3 chat format: the BOS token its template starts
    /// with, the default system turn where the template writes one, each turn, and the assistant
    /// header so that generation starts inside the assistant turn.
    pub(crate) fn messages_to_hermes3_prompt(messages: &[Message]) -> String {
        let mut prompt = String::new();
        prompt.push_str("<|begin_of_text|>");
        // The template's injection sits inside its own loop over the messages, so an empty
        // conversation is left alone rather than given a system turn to itself.
        if messages
            .first()
            .is_some_and(|first| !matches!(first, Message::System { .. }))
        {
            prompt.push_str(HERMES3_DEFAULT_SYSTEM_TURN);
        }

        for (index, message) in messages.iter().enumerate() {
            match message {
                Message::System { content, .. } => {
                    push_hermes3_turn(&mut prompt, "system", content.as_ref());
                }
                Message::User { content, .. } => {
                    push_hermes3_turn(&mut prompt, "user", content.as_ref());
                }
                Message::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    if tool_calls.is_empty() {
                        push_hermes3_turn(&mut prompt, "assistant", content.as_ref());
                    } else {
                        push_hermes3_tool_call_turn(&mut prompt, tool_calls);
                    }
                }
                Message::Tool { content, .. } => {
                    push_hermes3_tool_response(
                        &mut prompt,
                        content.as_ref(),
                        messages[..index].last(),
                        messages.get(index + 1),
                    );
                }
            }
        }

        push_hermes3_header(&mut prompt, "assistant");
        prompt
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, ToSchema)]
#[serde(untagged)]
pub enum MessageContent {
    /// The text contents of the message.
    #[serde(rename(serialize = "text", deserialize = "text"))]
    Text(String),
    /// An array of content parts with a defined type, each can be of type text or image_url when
    /// passing in images. You can pass multiple images by adding multiple image_url content
    /// parts. Image input is only supported when using the gpt-4o model.
    #[serde(rename(serialize = "array", deserialize = "array"))]
    Array(Vec<MessageContentPart>),
}

impl std::fmt::Display for MessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageContent::Text(text) => write!(f, "{}", text),
            MessageContent::Array(parts) => {
                let mut content = String::new();
                for part in parts {
                    content.push_str(&format!("{}\n", part))
                }
                write!(f, "{}", content)
            }
        }
    }
}

// We manually implement Deserialize here for more control.
impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Value = Value::deserialize(deserializer)?;

        if let Some(s) = value.as_str() {
            return Ok(MessageContent::Text(s.to_string()));
        }

        if let Some(arr) = value.as_array() {
            let parts: Result<Vec<MessageContentPart>, _> = arr
                .iter()
                .map(|v| serde_json::from_value(v.clone()).map_err(serde::de::Error::custom))
                .collect();
            return Ok(MessageContent::Array(parts?));
        }

        Err(serde::de::Error::custom(
            "Expected a string or an array of content parts",
        ))
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum MessageContentPart {
    #[serde(rename(serialize = "text", deserialize = "text"))]
    Text {
        /// The type of the content part.
        #[serde(rename(serialize = "type", deserialize = "type"))]
        r#type: String,
        /// The text content.
        text: String,
    },
    #[serde(rename(serialize = "image", deserialize = "image"))]
    Image {
        /// The type of the content part.
        #[serde(rename(serialize = "type", deserialize = "type"))]
        r#type: String,
        image_url: MessageContentPartImageUrl,
    },
}

impl std::fmt::Display for MessageContentPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageContentPart::Text { r#type, text } => {
                write!(f, "{}: {}", r#type, text)
            }
            MessageContentPart::Image { r#type, image_url } => {
                write!(f, "{}: [Image URL: {}]", r#type, image_url)
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "image_url", deserialize = "image_url"))]
pub struct MessageContentPartImageUrl {
    /// Either a URL of the image or the base64 encoded image data.
    url: String,
    /// Specifies the detail level of the image.
    detail: Option<String>,
}

/// Implementing Display for MessageContentPartImageUrl
impl std::fmt::Display for MessageContentPartImageUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.detail {
            Some(detail) => write!(f, "Image URL: {}, Detail: {}", self.url, detail),
            None => write!(f, "Image URL: {}", self.url),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ToolCallFunction {
    /// The name of the function to call.
    name: String,
    /// The arguments to call the function with, as generated by the model in JSON format.
    /// Note that the model does not always generate valid JSON, and may hallucinate parameters not
    /// defined by your function schema. Validate the arguments in your code before calling
    /// your function.
    arguments: Value,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "tool_call", deserialize = "tool_call"))]
pub struct ToolCall {
    /// The ID of the tool call.
    id: String,
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename(serialize = "type", deserialize = "type"))]
    r#type: String,
    /// The function that the model called.
    function: ToolCallFunction,
}

/// Writes JSON as Python's `json.dumps` writes it: a space after every colon and after every
/// comma, where `serde_json` writes neither.
///
/// A chat template renders a tool call's arguments through Jinja's `tojson`, which is
/// `json.dumps` with `ensure_ascii=False`. Its spacing is part of the prompt, so a server that
/// writes the compact form sends the model a different token sequence for the same tool call.
///
/// Floats are written in Python's notation for the same reason.
struct JsonDumpsFormatter;

/// Rewrites a float `serde_json` has written into the notation `json.dumps` uses, which is
/// Python's `repr`. The digits are left as they are.
///
/// `repr` writes an exponent of two digits at least, and turns to scientific notation below
/// `1e-4` where `serde_json` waits until `1e-5`.
fn python_float_notation(written: &str) -> String {
    if let Some((mantissa, exponent)) = written.split_once('e') {
        let (sign, digits) = match exponent.strip_prefix('-') {
            Some(digits) => ('-', digits),
            None => ('+', exponent.trim_start_matches('+')),
        };
        return format!("{mantissa}e{sign}{digits:0>2}");
    }

    let (sign, magnitude) = match written.strip_prefix('-') {
        Some(magnitude) => ("-", magnitude),
        None => ("", written),
    };
    let Some(fraction) = magnitude.strip_prefix("0.0000") else {
        return written.to_string();
    };
    let zeros = fraction.len() - fraction.trim_start_matches('0').len();
    let mut digits = fraction[zeros..].chars();
    let Some(first) = digits.next() else {
        return written.to_string();
    };
    let rest = digits.as_str();
    let point = if rest.is_empty() { "" } else { "." };
    format!("{sign}{first}{point}{rest}e-{:02}", zeros + 5)
}

/// Writes one float in `repr`'s notation, from what `serde_json` wrote for it.
fn write_python_float<W>(writer: &mut W, written: &[u8]) -> std::io::Result<()>
where
    W: ?Sized + std::io::Write,
{
    // serde_json writes a float as ASCII digits, so nothing is lost in the conversion.
    let written = String::from_utf8_lossy(written);
    writer.write_all(python_float_notation(&written).as_bytes())
}

impl Formatter for JsonDumpsFormatter {
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        let mut written = Vec::new();
        CompactFormatter.write_f64(&mut written, value)?;
        write_python_float(writer, &written)
    }

    fn write_f32<W>(&mut self, writer: &mut W, value: f32) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        let mut written = Vec::new();
        CompactFormatter.write_f32(&mut written, value)?;
        write_python_float(writer, &written)
    }
}

/// Renders `value` the way a chat template's `tojson` renders it.
///
/// The keys come out in the order the caller sent them, which is `preserve_order` on this
/// crate's `serde_json`: `json.dumps` writes a dict in its own order, and without the feature a
/// [`Value`] would sort them.
fn json_dumps(value: &Value) -> String {
    let mut written = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut written, JsonDumpsFormatter);
    value
        .serialize(&mut serializer)
        .expect("a Value serializes into a Vec without failing");
    String::from_utf8(written).expect("serde_json writes UTF-8")
}

impl ToolCall {
    pub fn function_call_string(&self, model: Model) -> String {
        match model {
            Model::HermesLlama318b | Model::HermesLlama3170b | Model::HermesLlama31405b => {
                // The template writes the name before the arguments.
                format!(
                    "{{\"name\": \"{}\", \"arguments\": {}}}",
                    self.function.name,
                    json_dumps(&self.function.arguments)
                )
            }
            Model::Llama38b
            | Model::Llama38bInstruct
            | Model::Llama370b
            | Model::Llama370bInstruct
            | Model::Llama31405b
            | Model::Llama31405bInstruct
            | Model::Llama318b
            | Model::Llama318bInstruct
            | Model::Llama3170b
            | Model::Llama3170bInstruct
            | Model::Llama321b
            | Model::Llama321bInstruct
            | Model::Llama323b
            | Model::Llama323bInstruct
            | Model::NousLlama318bInstruct => {
                // Check if arguments is a JSON object
                if let Some(args) = self.function.arguments.as_object() {
                    let params_str = args
                        .iter()
                        .map(|(k, v)| match v {
                            serde_json::Value::String(s) => format!("{}='{}'", k, s),
                            serde_json::Value::Number(n) => format!("{}={}", k, n),
                            serde_json::Value::Bool(b) => format!("{}={}", k, b),
                            _ => format!("{}={}", k, v),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{}({})", self.function.name, params_str)
                }
                // Check if arguments is a string (e.g., serialized JSON)
                else if let Some(args_str) = self.function.arguments.as_str() {
                    // Attempt to parse the string as JSON
                    if let Ok(serde_json::Value::Object(args)) =
                        serde_json::from_str::<serde_json::Value>(args_str)
                    {
                        let params_str = args
                            .iter()
                            .map(|(k, v)| match v {
                                serde_json::Value::String(s) => format!("{}='{}'", k, s),
                                serde_json::Value::Number(n) => format!("{}={}", k, n),
                                serde_json::Value::Bool(b) => format!("{}={}", k, b),
                                _ => format!("{}={}", k, v),
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{}({})", self.function.name, params_str)
                    } else {
                        // If parsing fails, include arguments as-is
                        format!("{}({})", self.function.name, args_str)
                    }
                } else {
                    // If arguments is neither an object nor a string, include function name only
                    format!("{}()", self.function.name)
                }
            }
            Model::Llama27b | Model::Llama27bChatHf | Model::Llama270b => {
                self.function.name.to_string()
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "tool", deserialize = "tool"))]
pub struct Tool {
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename(serialize = "type", deserialize = "type"))]
    r#type: String,
    /// The function that the model called.
    function: ToolFunction,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ToolFunction {
    /// Description of the function to call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    /// The name of the function to call.
    name: String,
    /// The arguments to call the function with, as generated by the model in JSON format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
    /// Whether to enable strict schema adherence when generating the function call. If set to
    /// true, the model will follow the exact schema defined in the parameters field. Only a
    /// subset of JSON Schema is supported when strict is true
    #[serde(default, skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

/// The stop condition for the chat completion.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "stop", deserialize = "stop"))]
#[serde(untagged)]
pub enum StopCondition {
    Array(Vec<String>),
    String(String),
}

/// What a chat completion request asks for.
///
/// Every field the API serves is declared here, and a body carrying one that is not is refused
/// rather than served as though it were absent: an unserved parameter and a misspelt budget both
/// change what the caller gets back, and dropping either in silence hides that.
#[derive(Debug, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename(serialize = "requestBody", deserialize = "requestBody"))]
#[serde(deny_unknown_fields)]
pub struct RequestBody {
    /// A list of messages comprising the conversation so far.
    messages: Vec<Message>,
    /// ID of the model to use.
    model: Model,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    /// frequency in the text so far, decreasing the model's likelihood to repeat the same line
    /// verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Accepts a JSON object that maps tokens (specified as their token ID in the tokenizer) to an
    /// associated bias value from -100 to 100.
    logit_bias: Option<HashMap<String, f32>>,
    /// Whether to return log probabilities of the output tokens or not. If true, returns the log
    /// probabilities of each output token returned in the content of message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    /// An integer between 0 and 20 specifying the number of most likely tokens to return at each
    /// token position, each with an associated log probability. logprobs must be set to true
    /// if this parameter is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<i32>,
    /// An upper bound for the number of tokens that can be generated for a completion,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    /// The deprecated name for `max_completion_tokens`, and still what most clients send. Read
    /// when `max_completion_tokens` is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// How many chat completion choices to generate for each input message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    /// A seed to use for random number generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will
    /// not contain the stop sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stop: Option<StopCondition>,
    /// If set, the server will stream the results as they come in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
    /// output more random, while lower values like 0.2 will make it more focused and
    /// deterministic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model
    /// considers the results of the tokens with top_p probability mass. So 0.1 means only the
    /// tokens comprising the top 10% probability mass are considered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// A list of tools the model may call. Currently, only functions are supported as a tool. Use
    /// this to provide a list of functions the model may generate JSON inputs for. A max of 128
    /// functions are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    /// A unique identifier representing your end-user, which can help the system to monitor and
    /// detect abuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

impl RequestBody {
    pub fn model(&self) -> &Model {
        &self.model
    }
}

/// What the engine is asked on a chat completion's behalf, before the prompt is tokenized.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineRequest {
    /// The conversation rendered through the model's chat template.
    pub prompt: String,
    pub sampling: SamplingParams,
    /// The most tokens to generate when the request bounds it. When it does not, the bound is
    /// the room left under the max model length, which is known only once the prompt is
    /// tokenized.
    pub max_new_tokens: Option<TokenCount>,
    /// Strings that end generation where they appear; matched after detokenization.
    pub stop: Vec<String>,
    pub stream: bool,
}

/// A request the engine cannot serve as asked. Every variant answers with a 400.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Refused {
    #[error("logprobs are not supported")]
    Logprobs,
    #[error("top_logprobs is not supported")]
    TopLogprobs,
    #[error("n must be 1; {n} choices are not supported")]
    Choices { n: usize },
    #[error("logit_bias is not supported")]
    LogitBias,
    #[error("tools are not supported")]
    Tools,
    #[error("frequency_penalty is not applied; it must be 0 or unset, not {frequency_penalty}")]
    FrequencyPenalty { frequency_penalty: f32 },
    #[error("presence_penalty is not applied; it must be 0 or unset, not {presence_penalty}")]
    PresencePenalty { presence_penalty: f32 },
    #[error("temperature must be between 0 and 2, not {temperature}")]
    Temperature { temperature: f32 },
    #[error("top_p must be between 0 and 1, not {top_p}")]
    TopP { top_p: f32 },
    #[error("max_completion_tokens must be at least 1")]
    ZeroCompletionTokens,
    #[error("max_tokens must be at least 1")]
    ZeroMaxTokens,
}

impl RequestBody {
    /// The engine request this body asks for, or why it cannot be served.
    ///
    /// Sampling is on, as the API's default is the model's own distribution; a temperature of
    /// zero is the request for greedy decoding, honoured by the sampler. `seed` is used when
    /// given and `fresh_seed` otherwise, so an unseeded request is not reproducible by
    /// accident. The completion budget is `max_completion_tokens`, or the deprecated
    /// `max_tokens` when that is absent. `user` is the caller's own identifier and is accepted
    /// without being acted on.
    ///
    /// # Errors
    ///
    /// Returns [`Refused`] for what the engine does not serve — `logprobs`, `top_logprobs`,
    /// more than one choice, `logit_bias`, `tools`, a non-zero `frequency_penalty` or
    /// `presence_penalty` — and for a temperature outside 0 to 2, a `top_p` outside 0 to 1, or
    /// a completion budget of zero under either of its names.
    pub fn to_engine_request(&self, fresh_seed: u64) -> Result<EngineRequest, Refused> {
        self.refuse_unserved()?;
        Ok(EngineRequest {
            prompt: self.model.messages_to_prompt(&self.messages),
            sampling: self.sampling(fresh_seed)?,
            max_new_tokens: self.max_new_tokens()?,
            stop: self.stop_strings(),
            stream: self.stream.unwrap_or(false),
        })
    }

    /// Refuses what the engine does not serve at all, when set to anything but its unset value.
    fn refuse_unserved(&self) -> Result<(), Refused> {
        if self.logprobs == Some(true) {
            return Err(Refused::Logprobs);
        }
        if self.top_logprobs.is_some() {
            return Err(Refused::TopLogprobs);
        }
        if let Some(n) = self.n.filter(|&n| n != 1) {
            return Err(Refused::Choices { n });
        }
        if self
            .logit_bias
            .as_ref()
            .is_some_and(|bias| !bias.is_empty())
        {
            return Err(Refused::LogitBias);
        }
        if self.tools.as_ref().is_some_and(|tools| !tools.is_empty()) {
            return Err(Refused::Tools);
        }
        Ok(())
    }

    /// The sampling parameters the engine applies, refusing those out of range and the
    /// penalties the sampler does not apply.
    fn sampling(&self, fresh_seed: u64) -> Result<SamplingParams, Refused> {
        let temperature = self.temperature.unwrap_or(1.0);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(Refused::Temperature { temperature });
        }
        let top_p = self.top_p.unwrap_or(1.0);
        if !(0.0..=1.0).contains(&top_p) {
            return Err(Refused::TopP { top_p });
        }
        if let Some(frequency_penalty) = self.frequency_penalty.filter(|&penalty| penalty != 0.0) {
            return Err(Refused::FrequencyPenalty { frequency_penalty });
        }
        if let Some(presence_penalty) = self.presence_penalty.filter(|&penalty| penalty != 0.0) {
            return Err(Refused::PresencePenalty { presence_penalty });
        }
        Ok(SamplingParams {
            temperature,
            top_p,
            do_sample: true,
            seed: self.seed.unwrap_or(fresh_seed),
            ..SamplingParams::default()
        })
    }

    /// The completion budget the request carries, under whichever name: `max_completion_tokens`
    /// wins when both are sent. `None` leaves the bound to the room under the max model length.
    fn max_new_tokens(&self) -> Result<Option<TokenCount>, Refused> {
        let (budget, refusal) = match (self.max_completion_tokens, self.max_tokens) {
            (Some(budget), _) => (budget, Refused::ZeroCompletionTokens),
            (None, Some(budget)) => (budget, Refused::ZeroMaxTokens),
            (None, None) => return Ok(None),
        };
        TokenCount::new(budget as usize).map(Some).ok_or(refusal)
    }

    fn stop_strings(&self) -> Vec<String> {
        match &self.stop {
            Some(StopCondition::Array(strings)) => strings.clone(),
            Some(StopCondition::String(string)) => vec![string.clone()],
            None => Vec::new(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub logprobs: Option<Value>,
    pub finish_reason: FinishReason,
}

/// Why a completion ended, in the API's own words: the model stopped, or the budget ran out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token, or a stop string was matched.
    Stop,
    /// The completion budget, or the max model length, was reached.
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    /// The usage of `prompt_tokens` and `completion_tokens`.
    #[must_use]
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        // A count past u32 is not a count this API will ever report; saturating names that
        // rather than wrapping.
        let prompt_tokens = u32::try_from(prompt_tokens).unwrap_or(u32::MAX);
        let completion_tokens = u32::try_from(completion_tokens).unwrap_or(u32::MAX);
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

impl From<EngineUsage> for Usage {
    fn from(usage: EngineUsage) -> Self {
        Self::new(usage.prompt_tokens, usage.generated_tokens)
    }
}

/// What this server reports as its configuration fingerprint.
pub const SYSTEM_FINGERPRINT: &str = concat!("atoma-infer/", env!("CARGO_PKG_VERSION"));

/// What identifies one completion in its response and in every chunk of its stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionIdentity {
    pub id: String,
    pub model: String,
    /// Seconds since the Unix epoch when the request arrived.
    pub created: u64,
}

impl ChatCompletionResponse {
    /// The one-choice response to a completed request.
    #[must_use]
    pub fn completed(
        identity: CompletionIdentity,
        content: String,
        finish_reason: FinishReason,
        usage: Usage,
    ) -> Self {
        let CompletionIdentity { id, model, created } = identity;
        Self {
            id,
            object: "chat.completion".into(),
            created,
            model,
            system_fingerprint: SYSTEM_FINGERPRINT.into(),
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(MessageContent::Text(content)),
                    name: None,
                    refusal: None,
                    tool_calls: vec![],
                },
                logprobs: None,
                finish_reason,
            }],
            usage,
        }
    }
}

/// Represents a chunk of a streaming chat completion response.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionChunk {
    /// A unique identifier for this chat completion.
    pub id: String,
    /// The object type, which is "chat.completion" for this struct.
    pub object: String,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The model used for this chat completion.
    pub model: String,
    /// A unique identifier for the model's configuration and version.
    pub system_fingerprint: String,
    /// An array of chat completion choices. Each choice represents a possible completion for the
    /// input.
    pub choices: Vec<StreamChoice>,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

/// Represents a single choice in a streaming chat completion response.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct StreamChoice {
    /// The index of this choice in the list of choices.
    pub index: u32,
    /// The delta (incremental update) for this choice.
    pub delta: Delta,
    /// Log probabilities for the tokens in this choice, if requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    /// The reason why the model stopped generating tokens, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Represents the delta (incremental update) in a streaming chat completion response.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct Delta {
    /// The role of the message author (e.g., "assistant").
    pub role: String,
    /// The content of the message, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// A refusal message, if the assistant refuses to respond.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// A list of tool calls made by the assistant, if any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

/// The chunks a streamed request is made of: the text as it comes, then the finish.
impl ChatCompletionChunk {
    /// A chunk carrying new text and nothing else.
    #[must_use]
    pub fn text(identity: &CompletionIdentity, content: String) -> Self {
        Self::chunk(identity, Some(content), None, Usage::new(0, 0))
    }

    /// The last chunk: no text, the finish reason and the usage.
    #[must_use]
    pub fn finished(
        identity: &CompletionIdentity,
        finish_reason: FinishReason,
        usage: Usage,
    ) -> Self {
        Self::chunk(identity, None, Some(finish_reason), usage)
    }

    fn chunk(
        identity: &CompletionIdentity,
        content: Option<String>,
        finish_reason: Option<FinishReason>,
        usage: Usage,
    ) -> Self {
        Self {
            id: identity.id.clone(),
            object: "chat.completion.chunk".into(),
            created: identity.created,
            model: identity.model.clone(),
            system_fingerprint: SYSTEM_FINGERPRINT.into(),
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: "assistant".into(),
                    content,
                    refusal: None,
                    tool_calls: vec![],
                },
                logprobs: None,
                finish_reason,
            }],
            usage,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use serde_json::json;

    use atoma_core::request::SamplingParams;
    use atoma_core::types::TokenCount;

    use super::{
        json_dumps, messages, ChatCompletionChunk, ChatCompletionResponse, Choice,
        CompletionIdentity, Delta, FinishReason, Message, MessageContent, MessageContentPart,
        MessageContentPartImageUrl, Model, Refused, RequestBody, StreamChoice, ToolCall,
        ToolCallFunction, Usage,
    };

    fn identity(created: u64) -> CompletionIdentity {
        CompletionIdentity {
            id: "chatcmpl-1".into(),
            model: "llama".into(),
            created,
        }
    }

    #[test]
    fn deserialize_request_body_basic() {
        let json_request_body = r#"
            {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant"
                    }
                ]
            }
        "#;

        let request_body: Result<RequestBody, serde_json::Error> =
            serde_json::from_str(json_request_body);
        assert!(request_body.is_ok());
    }

    /// The benchmark harness holds the sampler still by sending `temperature: 0`, and asks for a
    /// token budget with `max_completion_tokens` rather than the deprecated `max_tokens`. Both have
    /// to survive the hop into `GenerateParameters` for the harness to measure this engine at all:
    /// a dropped budget generates until the model's own stop condition, and a rejected temperature
    /// refuses the request outright. Mirrors `completion_body` in `crates/bench/src/client.rs`.
    #[test]
    fn benchmark_harness_body_reaches_the_engine_intact() {
        let json_request_body = r#"
            {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "messages": [{ "role": "user", "content": "Hello" }],
                "max_completion_tokens": 128,
                "stream": true,
                "temperature": 0.0
            }
        "#;

        let request_body: RequestBody =
            serde_json::from_str(json_request_body).expect("Harness body must deserialize");
        let request = request_body.to_engine_request(7).unwrap();

        assert_eq!(request.sampling.temperature, 0.0);
        assert!(request.sampling.do_sample);
        assert_eq!(request.max_new_tokens, Some(TokenCount::new(128).unwrap()));
        assert!(request.stream);
        assert!(request.prompt.contains("Hello"));
        assert!(request.stop.is_empty());
    }

    /// A field the API does not declare changes what the caller gets back — a misspelt budget
    /// lets generation run to the model's own stop — so the body is refused rather than served
    /// with the field dropped.
    #[test]
    fn a_field_the_api_does_not_declare_is_refused_by_name() {
        let error = serde_json::from_value::<RequestBody>(json!({
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": [{ "role": "user", "content": "Hi" }],
            "max_completion_token": 8
        }))
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unknown field `max_completion_token`"),
            "{error}"
        );
    }

    fn body(fields: serde_json::Value) -> RequestBody {
        let mut body = json!({
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": [{ "role": "user", "content": "Hi" }]
        });
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn a_body_that_asks_for_nothing_samples_from_the_models_own_distribution() {
        let request = body(json!({})).to_engine_request(42).unwrap();
        assert_eq!(
            request.sampling,
            SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                do_sample: true,
                seed: 42,
                ..SamplingParams::default()
            }
        );
        assert_eq!(
            request.max_new_tokens, None,
            "the room left under the max model length"
        );
        assert!(!request.stream);
        assert!(request.stop.is_empty());
    }

    #[test]
    fn a_seed_given_is_kept_and_one_absent_is_the_fresh_one() {
        assert_eq!(
            body(json!({ "seed": 9 }))
                .to_engine_request(42)
                .unwrap()
                .sampling
                .seed,
            9
        );
        assert_eq!(
            body(json!({})).to_engine_request(42).unwrap().sampling.seed,
            42
        );
    }

    /// `max_tokens` is the deprecated name for the budget and still what most clients send; a
    /// request carrying it alone is bounded by it rather than served as though no budget were set.
    #[test]
    fn a_budget_sent_as_max_tokens_is_applied() {
        let request = body(json!({ "max_tokens": 8 }))
            .to_engine_request(1)
            .unwrap();
        assert_eq!(request.max_new_tokens, Some(TokenCount::new(8).unwrap()));
    }

    #[test]
    fn max_completion_tokens_wins_when_both_budgets_are_sent() {
        let request = body(json!({ "max_completion_tokens": 3, "max_tokens": 8 }))
            .to_engine_request(1)
            .unwrap();
        assert_eq!(request.max_new_tokens, Some(TokenCount::new(3).unwrap()));
    }

    #[test]
    fn stop_strings_arrive_as_one_or_several() {
        assert_eq!(
            body(json!({ "stop": "\n" }))
                .to_engine_request(1)
                .unwrap()
                .stop,
            ["\n"]
        );
        assert_eq!(
            body(json!({ "stop": ["a", "b"] }))
                .to_engine_request(1)
                .unwrap()
                .stop,
            ["a", "b"]
        );
    }

    #[test]
    fn what_the_engine_cannot_honour_is_refused_by_name() {
        let refused = |fields: serde_json::Value| body(fields).to_engine_request(1).unwrap_err();
        assert_eq!(refused(json!({ "logprobs": true })), Refused::Logprobs);
        assert_eq!(refused(json!({ "top_logprobs": 2 })), Refused::TopLogprobs);
        assert_eq!(refused(json!({ "n": 2 })), Refused::Choices { n: 2 });
        assert_eq!(
            refused(json!({ "logit_bias": { "50256": -100 } })),
            Refused::LogitBias
        );
        assert_eq!(
            refused(json!({ "tools": [{ "type": "function", "function": { "name": "f" } }] })),
            Refused::Tools
        );
        assert_eq!(
            refused(json!({ "frequency_penalty": 0.5 })),
            Refused::FrequencyPenalty {
                frequency_penalty: 0.5
            }
        );
        assert_eq!(
            refused(json!({ "presence_penalty": -0.5 })),
            Refused::PresencePenalty {
                presence_penalty: -0.5
            }
        );
        assert_eq!(
            refused(json!({ "temperature": 2.5 })),
            Refused::Temperature { temperature: 2.5 }
        );
        assert_eq!(
            refused(json!({ "top_p": 1.5 })),
            Refused::TopP { top_p: 1.5 }
        );
        assert_eq!(
            refused(json!({ "max_completion_tokens": 0 })),
            Refused::ZeroCompletionTokens
        );
        assert_eq!(refused(json!({ "max_tokens": 0 })), Refused::ZeroMaxTokens);
        assert!(
            refused(json!({ "n": 3 })).to_string().contains("3 choices"),
            "the refusal names what was asked"
        );
    }

    #[test]
    fn the_unset_values_of_refusable_fields_are_accepted() {
        assert!(
            body(json!({ "logprobs": false, "n": 1, "logit_bias": {}, "tools": [] }))
                .to_engine_request(1)
                .is_ok()
        );
        assert!(body(json!({
            "temperature": 0,
            "top_p": 0,
            "frequency_penalty": 0,
            "presence_penalty": 0.0,
            "user": "someone"
        }))
        .to_engine_request(1)
        .is_ok());
    }

    #[test]
    fn deserialize_system_message() {
        let json_system_message_text = r#"
            {
                "role": "system",
                "content": "Hello, World!"
            }
        "#;

        let system_message: Result<Message, serde_json::Error> =
            serde_json::from_str(json_system_message_text);
        assert_eq!(
            system_message.unwrap(),
            Message::System {
                content: Some(MessageContent::Text("Hello, World!".to_string())),
                name: None
            }
        );
    }

    #[test]
    fn deserialize_user_message() {
        let json_user_message_text = r#"
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, World!"
                    },
                    {
                        "type": "image",
                        "image_url": {
                            "url": "http://example.com/image.png",
                            "detail": "high"
                        }
                    }
                ]
            }
        "#;

        let user_message: Result<Message, serde_json::Error> =
            serde_json::from_str(json_user_message_text);
        assert_eq!(
            user_message.unwrap(),
            Message::User {
                content: Some(MessageContent::Array(vec![
                    MessageContentPart::Text {
                        r#type: "text".into(),
                        text: "Hello, World!".into(),
                    },
                    MessageContentPart::Image {
                        r#type: "image".into(),
                        image_url: MessageContentPartImageUrl {
                            url: "http://example.com/image.png".into(),
                            detail: Some("high".into()),
                        },
                    },
                ])),
                name: None,
            }
        );
    }

    #[test]
    fn deserialize_assistant_message() {
        let json_assistant_message_text = r#"
            {
                "role": "assistant",
                "content": "Sure! Here is your answer: ...",
                "refusal": null,
                "tool_calls": [{
                    "id": "chatcmpl-123",
                    "type": "function",
                    "function": {
                        "name": "myFunction",
                        "arguments": {
                            "key": "value"
                        }
                    }
                }]
            }
        "#;

        let assistant_message: Result<Message, serde_json::Error> =
            serde_json::from_str(json_assistant_message_text);
        assert_eq!(
            assistant_message.unwrap(),
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Sure! Here is your answer: ...".to_string()
                )),
                name: None,
                refusal: None,
                tool_calls: vec![ToolCall {
                    id: "chatcmpl-123".into(),
                    r#type: "function".into(),
                    function: ToolCallFunction {
                        name: "myFunction".into(),
                        arguments: json!({"key": "value"}),
                    },
                }],
            }
        );
    }

    #[test]
    fn deserialize_tool_message() {
        let json_tool_message_text = r#"
            {
                "role": "tool",
                "content": "Using tool ...",
                "tool_call_id": "123"
            }
        "#;

        let tool_message: Result<Message, serde_json::Error> =
            serde_json::from_str(json_tool_message_text);
        assert_eq!(
            tool_message.unwrap(),
            Message::Tool {
                content: Some(MessageContent::Text("Using tool ...".to_string())),
                tool_call_id: "123".into(),
            }
        );
    }

    #[test]
    fn deserialize_message_content_text() {
        let json_message_content_text = r#"
            "Hello, World!"
        "#;

        let message_content: Result<MessageContent, serde_json::Error> =
            serde_json::from_str(json_message_content_text);
        assert!(message_content.is_ok());
    }

    #[test]
    fn deserialize_message_content_array() {
        let json_message_content_array = r#"
            [
                {
                    "type": "text",
                    "text": "Hello, World!"
                },
                {
                    "type": "image",
                    "image_url": {
                        "url": "http://example.com/image.png",
                        "detail": "high"
                    }
                }
            ]
        "#;

        let message_content: Result<MessageContent, serde_json::Error> =
            serde_json::from_str(json_message_content_array);
        assert!(message_content.is_ok());
    }

    #[test]
    fn test_empty_prompt() {
        let messages: Vec<Message> = vec![];
        let result = messages::messages_to_llama3_prompt(&messages);
        assert_eq!(
            result,
            "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn test_system_message_only() {
        let messages = vec![Message::System {
            content: Some(MessageContent::Text(
                "You are a helpful assistant.".to_string(),
            )),
            name: None,
        }];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\n",
            "You are a helpful assistant.<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_user_message_only() {
        let messages = vec![Message::User {
            content: Some(MessageContent::Text("Hello, who are you?".to_string())),
            name: None,
        }];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "Hello, who are you?<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_assistant_message_only() {
        let messages = vec![Message::Assistant {
            content: Some(MessageContent::Text("I am an AI assistant.".to_string())),
            name: None,
            refusal: None,
            tool_calls: vec![],
        }];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "I am an AI assistant.<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tool_message_only() {
        let messages = vec![Message::Tool {
            content: Some(MessageContent::Text("25 C".to_string())),
            tool_call_id: "get_weather".to_string(),
        }];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>ipython<|end_header_id|>\n\n",
            "25 C<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_system_and_user() {
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text(
                    "You are a helpful assistant.".to_string(),
                )),
                name: None,
            },
            Message::User {
                content: Some(MessageContent::Text("Hello, who are you?".to_string())),
                name: None,
            },
        ];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\n",
            "You are a helpful assistant.<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "Hello, who are you?<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_system_and_assistant() {
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text(
                    "You are a helpful assistant.".to_string(),
                )),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("I am an AI assistant.".to_string())),
                name: None,
                refusal: None,
                tool_calls: vec![],
            },
        ];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\n",
            "You are a helpful assistant.<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "I am an AI assistant.<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_user_and_assistant() {
        let messages = vec![
            Message::User {
                content: Some(MessageContent::Text("Hello, who are you?".to_string())),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("I am an AI assistant.".to_string())),
                name: None,
                refusal: None,
                tool_calls: vec![],
            },
        ];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "Hello, who are you?<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "I am an AI assistant.<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_system_user_assistant() {
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text(
                    "You are a helpful assistant.".to_string(),
                )),
                name: None,
            },
            Message::User {
                content: Some(MessageContent::Text(
                    "What is the weather in SF?".to_string(),
                )),
                name: None,
            },
            Message::Assistant {
                content: None,
                name: None,
                refusal: None,
                tool_calls: vec![ToolCall {
                    id: "get_weather".to_string(),
                    r#type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: json!({
                            "city": "San Francisco",
                            "metric": "celsius"
                        }),
                    },
                }],
            },
            Message::Tool {
                content: Some(MessageContent::Text("\"25 C\"".to_string())),
                tool_call_id: "get_weather".to_string(),
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "The weather in San Francisco is 25 C.".to_string(),
                )),
                name: None,
                refusal: None,
                tool_calls: vec![],
            },
        ];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            // System message
            "<|start_header_id|>system<|end_header_id|>\n\n",
            "You are a helpful assistant.<|eot_id|>",
            // User message
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "What is the weather in SF?<|eot_id|>",
            // Assistant message with tool call
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|python_tag|>[get_weather(city='San Francisco', metric='celsius')]<|eot_id|>",
            // Tool response
            "<|start_header_id|>ipython<|end_header_id|>\n\n",
            "\"25 C\"<|eot_id|>",
            // Assistant's final response
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "The weather in San Francisco is 25 C.<|eot_id|>",
            // The generation header
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tool_call_with_multiple_functions() {
        let messages = vec![Message::Assistant {
            content: None,
            name: None,
            refusal: None,
            tool_calls: vec![
                ToolCall {
                    id: "1".to_string(),
                    r#type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "func1".to_string(),
                        arguments: json!({
                            "param1": "value1"
                        }),
                    },
                },
                ToolCall {
                    id: "2".to_string(),
                    r#type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "func2".to_string(),
                        arguments: json!({
                            "param2": "value2"
                        }),
                    },
                },
            ],
        }];
        let result = messages::messages_to_llama3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|python_tag|>[func1(param1='value1'), func2(param2='value2')]<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(result, expected);
    }

    /// One conversation per trailing role, the assistant's two ways of speaking, and the empty
    /// one.
    fn every_trailing_role() -> [Vec<Message>; 6] {
        let text = |text: &str| Some(MessageContent::Text(text.to_string()));
        [
            vec![],
            vec![Message::System {
                content: text("Be brief."),
                name: None,
            }],
            vec![Message::User {
                content: text("Hi"),
                name: None,
            }],
            vec![Message::Assistant {
                content: text("Hello"),
                name: None,
                refusal: None,
                tool_calls: vec![],
            }],
            vec![Message::Assistant {
                content: None,
                name: None,
                refusal: None,
                tool_calls: vec![ToolCall {
                    id: "1".to_string(),
                    r#type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: serde_json::json!({ "city": "Lisbon" }),
                    },
                }],
            }],
            vec![Message::Tool {
                content: text("25 C"),
                tool_call_id: "1".to_string(),
            }],
        ]
    }

    /// Generation starts inside the assistant turn whatever the last message was: without the
    /// header the model writes one itself, and the role name is decoded and served as content.
    #[test]
    fn a_llama3_prompt_ends_in_the_assistant_header_whatever_the_trailing_role() {
        for messages in every_trailing_role() {
            let prompt = messages::messages_to_llama3_prompt(&messages);
            assert!(
                prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
                "{messages:?} rendered as {prompt:?}"
            );
        }
    }

    #[test]
    fn test_system_message() {
        let messages = vec![Message::System {
            content: Some(MessageContent::Text(
                "You are Hermes 3, a superintelligent AI.".to_string(),
            )),
            name: None,
        }];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are Hermes 3, a superintelligent AI.<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_user_message() {
        let messages = vec![Message::User {
            content: Some(MessageContent::Text("Hello, who are you?".to_string())),
            name: None,
        }];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_assistant_message() {
        let messages = vec![Message::Assistant {
            content: Some(MessageContent::Text(
                "I am Hermes 3, a superintelligent AI.".to_string(),
            )),
            name: None,
            refusal: None,
            tool_calls: vec![],
        }];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>assistant\nI am Hermes 3, a superintelligent AI.<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_tool_message() {
        let messages = vec![Message::Tool {
            content: Some(MessageContent::Text("Tool response here.".to_string())),
            tool_call_id: "tool_call_id".to_string(),
        }];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<tool_response>\nTool response here.\n</tool_response><|im_end|>",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_tool_call_in_assistant_message() {
        let tool_call = ToolCall {
            id: "1".to_string(),
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: "get_stock_fundamentals".to_string(),
                arguments: serde_json::json!({"symbol": "TSLA"}),
            },
        };

        let messages = vec![Message::Assistant {
            content: None,
            name: None,
            refusal: None,
            tool_calls: vec![tool_call],
        }];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>assistant",
            "\n<tool_call>\n",
            "{\"name\": \"get_stock_fundamentals\", \"arguments\": {\"symbol\": \"TSLA\"}}",
            "\n</tool_call>",
            "<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_mixed_messages() {
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text(
                    "You are Hermes 3, a superintelligent AI.".to_string(),
                )),
                name: None,
            },
            Message::User {
                content: Some(MessageContent::Text(
                    "Fetch stock data for TSLA.".to_string(),
                )),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("Fetching stock data...".to_string())),
                name: None,
                refusal: None,
                tool_calls: vec![],
            },
        ];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are Hermes 3, a superintelligent AI.<|im_end|>\n",
            "<|im_start|>user\nFetch stock data for TSLA.<|im_end|>\n",
            "<|im_start|>assistant\nFetching stock data...<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_hermes3_empty_messages() {
        let messages: Vec<Message> = vec![];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        assert_eq!(prompt, "<|begin_of_text|><|im_start|>assistant\n");
    }

    #[test]
    fn test_hermes3_missing_content_in_message() {
        let messages = vec![Message::User {
            content: None,
            name: None,
        }];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        // Missing content is an empty string.
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    /// The template's `tojson` is `json.dumps`, which spaces every colon and every comma and
    /// writes a dict in its own order. A single string-valued argument is the one shape a colon
    /// substitution also gets right, so the arguments here are the ones that told the two apart.
    /// Every object below is written out of alphabetical order, which a sorted map would not
    /// preserve.
    #[test]
    fn hermes3_tool_call_arguments_are_spaced_the_way_the_template_writes_them() {
        let call = |arguments: serde_json::Value| {
            ToolCall {
                id: "1".to_string(),
                r#type: "function".to_string(),
                function: ToolCallFunction {
                    name: "f".to_string(),
                    arguments,
                },
            }
            .function_call_string(Model::HermesLlama318b)
        };

        assert_eq!(
            call(json!({ "symbol": "TSLA", "shares": 5 })),
            "{\"name\": \"f\", \"arguments\": {\"symbol\": \"TSLA\", \"shares\": 5}}",
            "the caller's order, spaced after the colon whatever the value's type"
        );
        assert_eq!(
            call(json!({ "nested": { "a": 1 }, "list": [1, 2], "ok": true })),
            "{\"name\": \"f\", \"arguments\": \
             {\"nested\": {\"a\": 1}, \"list\": [1, 2], \"ok\": true}}"
        );
        assert_eq!(
            call(json!({})),
            "{\"name\": \"f\", \"arguments\": {}}",
            "an empty object carries no separator to space"
        );
        assert_eq!(
            call(json!({ "city": "Köln", "note": "中文 😀" })),
            "{\"name\": \"f\", \"arguments\": {\"city\": \"Köln\", \"note\": \"中文 😀\"}}",
            "the template's tojson passes ensure_ascii=False, so text is not escaped"
        );
    }

    /// `json.dumps` writes a float as Python's `repr` does, which `serde_json` departs from twice:
    /// it writes `0.00001` where `repr` turns to scientific notation, and pads no exponent.
    #[test]
    fn a_float_argument_is_written_as_the_template_writes_it() {
        let arguments = |arguments: serde_json::Value| {
            let call = ToolCall {
                id: "1".to_string(),
                r#type: "function".to_string(),
                function: ToolCallFunction {
                    name: "f".to_string(),
                    arguments,
                },
            };
            call.function_call_string(Model::HermesLlama318b)
        };

        assert_eq!(
            arguments(json!({ "a": 1e-5, "b": 1e-6 })),
            "{\"name\": \"f\", \"arguments\": {\"a\": 1e-05, \"b\": 1e-06}}",
            "the two values serde_json writes differently"
        );
        assert_eq!(
            arguments(json!({ "a": 0.0001, "b": 1.5, "c": 1e16, "d": -1.25e-7 })),
            "{\"name\": \"f\", \"arguments\": \
             {\"a\": 0.0001, \"b\": 1.5, \"c\": 1e+16, \"d\": -1.25e-07}}",
            "the magnitudes on either side of each boundary"
        );
        assert_eq!(
            arguments(json!({ "a": 1e15, "b": 1e-100, "c": 0.0, "d": 5e-324 })),
            "{\"name\": \"f\", \"arguments\": \
             {\"a\": 1000000000000000.0, \"b\": 1e-100, \"c\": 0.0, \"d\": 5e-324}}",
            "an exponent of three digits is not padded, and nothing else is rewritten"
        );
    }

    /// The digits are `serde_json`'s, whatever the notation: where two shortest representations
    /// round-trip, `repr` and `serde_json` pick the same one and `f64`'s own `Display` does not.
    #[test]
    fn a_float_argument_keeps_the_digits_serde_json_writes() {
        let value = f64::from_bits(0xc30a_a61f_a224_75ca);
        assert_eq!(format!("{value}"), "-937625523621561.3");
        assert_eq!(json_dumps(&json!(value)), "-937625523621561.2");
    }

    /// The Llama 3 tool call renders its arguments in the caller's order too, which is
    /// `preserve_order` rather than the alphabetical order a sorted map would have imposed.
    #[test]
    fn a_llama3_tool_call_keeps_the_callers_argument_order() {
        let call = ToolCall {
            id: "1".to_string(),
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: "get_weather".to_string(),
                arguments: json!({ "city": "Lisbon", "at": "noon" }),
            },
        };
        assert_eq!(
            call.function_call_string(Model::Llama318bInstruct),
            "get_weather(city='Lisbon', at='noon')"
        );
    }

    /// The template writes one `<tool_call>` block per call rather than one block holding every
    /// call, and opens the turn with the role alone: the newline a text turn carries after the
    /// role is the one each block opens with.
    #[test]
    fn test_hermes3_multiple_tool_calls() {
        let tool_call1 = ToolCall {
            id: "1".to_string(),
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: "get_stock_fundamentals".to_string(),
                arguments: serde_json::json!({"symbol": "TSLA"}),
            },
        };

        let tool_call2 = ToolCall {
            id: "2".to_string(),
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: "get_crypto_data".to_string(),
                arguments: serde_json::json!({"symbol": "BTC"}),
            },
        };

        let messages = vec![Message::Assistant {
            content: None,
            name: None,
            refusal: None,
            tool_calls: vec![tool_call1, tool_call2],
        }];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>assistant",
            "\n<tool_call>\n",
            "{\"name\": \"get_stock_fundamentals\", \"arguments\": {\"symbol\": \"TSLA\"}}",
            "\n</tool_call>",
            "\n<tool_call>\n",
            "{\"name\": \"get_crypto_data\", \"arguments\": {\"symbol\": \"BTC\"}}",
            "\n</tool_call>",
            "<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_hermes3_tool_message_with_tool_call_id() {
        let messages = vec![Message::Tool {
            content: Some(MessageContent::Text("Stock data for TSLA".to_string())),
            tool_call_id: "123".to_string(),
        }];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<tool_response>\nStock data for TSLA\n</tool_response><|im_end|>",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    /// The `tool_use` template frames a tool message in `<tool_response>` tags rather than writing
    /// its content as a turn. The role line comes from the previous message, so a conversation
    /// opening with a tool message gets none.
    #[test]
    fn a_hermes3_tool_message_is_framed_as_a_tool_response_block() {
        let user = || Message::User {
            content: Some(MessageContent::Text("Weather?".to_string())),
            name: None,
        };
        let tool = || Message::Tool {
            content: Some(MessageContent::Text("25 C".to_string())),
            tool_call_id: "1".to_string(),
        };

        assert_eq!(
            messages::messages_to_hermes3_prompt(&[user(), tool()]),
            concat!(
                "<|begin_of_text|>",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                "<|im_start|>user\nWeather?<|im_end|>\n",
                "<|im_start|>tool\n<tool_response>\n25 C\n</tool_response><|im_end|>",
                "<|im_start|>assistant\n",
            ),
            "a tool message after a message of another role opens a turn of its own"
        );
        assert_eq!(
            messages::messages_to_hermes3_prompt(&[tool(), user()]),
            concat!(
                "<|begin_of_text|>",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                "<tool_response>\n25 C\n</tool_response>\n<|im_end|>",
                "<|im_start|>user\nWeather?<|im_end|>\n",
                "<|im_start|>assistant\n",
            ),
            "a tool message with nothing before it gets no role line"
        );
    }

    /// A run of consecutive tool messages is one turn: one role line, one block per message, one
    /// end-of-turn token.
    #[test]
    fn consecutive_hermes3_tool_messages_are_one_turn() {
        let user = || Message::User {
            content: Some(MessageContent::Text("Weather?".to_string())),
            name: None,
        };
        let tool = |content: &str| Message::Tool {
            content: Some(MessageContent::Text(content.to_string())),
            tool_call_id: "1".to_string(),
        };

        assert_eq!(
            messages::messages_to_hermes3_prompt(&[user(), tool("25 C"), tool("Cloudy")]),
            concat!(
                "<|begin_of_text|>",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                "<|im_start|>user\nWeather?<|im_end|>\n",
                "<|im_start|>tool\n",
                "<tool_response>\n25 C\n</tool_response>\n",
                "<tool_response>\nCloudy\n</tool_response>",
                "<|im_end|>",
                "<|im_start|>assistant\n",
            ),
            "two blocks in one turn"
        );
        assert_eq!(
            messages::messages_to_hermes3_prompt(&[tool("25 C"), tool("Cloudy"), user()]),
            concat!(
                "<|begin_of_text|>",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                "<tool_response>\n25 C\n</tool_response>\n",
                "<tool_response>\nCloudy\n</tool_response>\n",
                "<|im_end|>",
                "<|im_start|>user\nWeather?<|im_end|>\n",
                "<|im_start|>assistant\n",
            ),
            "the run ends where the tool messages end"
        );
    }

    #[test]
    fn test_hermes3_system_and_user_message_no_content() {
        let messages = vec![
            Message::System {
                content: None,
                name: None,
            },
            Message::User {
                content: None,
                name: None,
            },
        ];

        let prompt = messages::messages_to_hermes3_prompt(&messages);
        let expected = concat!(
            "<|begin_of_text|>",
            "<|im_start|>system\n<|im_end|>\n",
            "<|im_start|>user\n<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(prompt, expected);
    }

    /// A conversation that does not open with a system message is given the template's own, so
    /// the model is addressed under the same system prompt on either server. The template writes
    /// it from inside its loop over the messages, so an empty conversation gets none.
    #[test]
    fn a_hermes3_conversation_is_given_the_templates_system_turn_where_it_writes_one() {
        let user = || Message::User {
            content: Some(MessageContent::Text("Hi".to_string())),
            name: None,
        };
        let system = || Message::System {
            content: Some(MessageContent::Text("Be brief.".to_string())),
            name: None,
        };

        assert_eq!(
            messages::messages_to_hermes3_prompt(&[user()]),
            concat!(
                "<|begin_of_text|>",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                "<|im_start|>user\nHi<|im_end|>\n",
                "<|im_start|>assistant\n",
            ),
            "written ahead of the conversation, not anywhere in it"
        );
        assert_eq!(
            messages::messages_to_hermes3_prompt(&[system(), user()]),
            concat!(
                "<|begin_of_text|>",
                "<|im_start|>system\nBe brief.<|im_end|>\n",
                "<|im_start|>user\nHi<|im_end|>\n",
                "<|im_start|>assistant\n",
            ),
            "a system message of the conversation's own is not doubled"
        );
        assert_eq!(
            messages::messages_to_hermes3_prompt(&[]),
            "<|begin_of_text|><|im_start|>assistant\n",
            "the template writes its system turn from inside its loop over the messages"
        );
    }

    /// The template writes `<|im_end|>` straight after a turn's content. The builder wrote a
    /// newline between the two, so every turn carried a token the model's own template does not.
    ///
    /// A tool run that something follows is the template's one exception, and no conversation
    /// here ends in one.
    #[test]
    fn a_hermes3_turn_ends_in_the_end_of_turn_token_with_nothing_before_it() {
        for messages in every_trailing_role() {
            let prompt = messages::messages_to_hermes3_prompt(&messages);
            assert!(
                !prompt.contains("\n<|im_end|>"),
                "{messages:?} rendered as {prompt:?}"
            );
        }
        assert!(
            messages::messages_to_hermes3_prompt(&[Message::User {
                content: Some(MessageContent::Text("Hi".to_string())),
                name: None,
            }])
            .contains("<|im_start|>user\nHi<|im_end|>\n"),
            "the content runs straight into the tag, rather than the tag being absent"
        );
    }

    /// The Hermes 3 template ends in `<|im_start|>assistant` and a newline under its generation
    /// prompt, so generation starts inside the assistant turn whatever the last message was.
    #[test]
    fn a_hermes3_prompt_ends_in_the_assistant_header_whatever_the_trailing_role() {
        for messages in every_trailing_role() {
            let prompt = messages::messages_to_hermes3_prompt(&messages);
            assert!(
                prompt.ends_with("<|im_start|>assistant\n"),
                "{messages:?} rendered as {prompt:?}"
            );
        }
    }

    #[test]
    fn test_messages_to_llama2_prompt() {
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text(
                    "You are a helpful assistant.".to_string(),
                )),
                name: None,
            },
            Message::User {
                content: Some(MessageContent::Text("Hello, how are you?".to_string())),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "I'm doing well, thank you! How can I assist you today?".to_string(),
                )),
                name: None,
                refusal: None,
                tool_calls: Vec::new(),
            },
            Message::User {
                content: Some(MessageContent::Text("Can you tell me a joke?".to_string())),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Sure! Why did the computer show up at work late? Because it had a hard drive!"
                        .to_string(),
                )),
                name: None,
                refusal: None,
                tool_calls: Vec::new(),
            },
        ];

        let model = Model::Llama27b;

        let prompt = model.messages_to_prompt(&messages);

        let expected_prompt = "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nHello, how are you? [/INST]\nI'm doing well, thank you! How can I assist you today?\n[INST] Can you tell me a joke? [/INST]\nSure! Why did the computer show up at work late? Because it had a hard drive!\n";

        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_empty_string_message() {
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text("".to_string())),
                name: None,
            },
            Message::User {
                content: Some(MessageContent::Text("".to_string())),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("".to_string())),
                name: None,
                refusal: None,
                tool_calls: Vec::new(),
            },
        ];

        let model = Model::Llama27b;

        let prompt = model.messages_to_prompt(&messages);

        let expected_prompt = "<s>[INST] <<SYS>>\n\n<</SYS>>\n\n [/INST]\n\n";

        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_no_system_message() {
        let messages = vec![
            Message::User {
                content: Some(MessageContent::Text(
                    "What is the weather like?".to_string(),
                )),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "The weather is sunny today.".to_string(),
                )),
                name: None,
                refusal: None,
                tool_calls: Vec::new(),
            },
        ];

        let model = Model::Llama27b;

        let prompt = model.messages_to_prompt(&messages);

        let expected_prompt =
            "<s>[INST] What is the weather like? [/INST]\nThe weather is sunny today.\n";

        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_only_system_and_assistant_messages() {
        let messages = vec![
            Message::System {
                content: Some(MessageContent::Text("You are an AI assistant.".to_string())),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Hello, how can I assist you today?".to_string(),
                )),
                name: None,
                refusal: None,
                tool_calls: Vec::new(),
            },
        ];

        let model = Model::Llama27b;

        let prompt = model.messages_to_prompt(&messages);

        let expected_prompt = "<s>[INST] <<SYS>>\nYou are an AI assistant.\n<</SYS>>\n\n[/INST]\nHello, how can I assist you today?\n";

        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_only_user_message() {
        let messages = vec![Message::User {
            content: Some(MessageContent::Text("Is the sky blue?".to_string())),
            name: None,
        }];

        let model = Model::Llama27b;

        let prompt = model.messages_to_prompt(&messages);

        let expected_prompt = "<s>[INST] Is the sky blue? [/INST]\n";

        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_only_system_message() {
        let messages = vec![Message::System {
            content: Some(MessageContent::Text(
                "You are a helpful AI assistant.".to_string(),
            )),
            name: None,
        }];

        let model = Model::Llama27b;

        let prompt = model.messages_to_prompt(&messages);

        let expected_prompt =
            "<s>[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n[/INST]\n";

        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_only_assistant_message() {
        let messages = vec![Message::Assistant {
            content: Some(MessageContent::Text(
                "You are a helpful AI assistant.".to_string(),
            )),
            name: None,
            refusal: None,
            tool_calls: Vec::new(),
        }];

        let model = Model::Llama27b;

        let prompt = model.messages_to_prompt(&messages);

        let expected_prompt = "<s>You are a helpful AI assistant.\n";

        assert_eq!(prompt, expected_prompt);
    }

    #[test]
    fn test_deserialize_chat_completion_response() {
        let json = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        });

        let response: ChatCompletionResponse = serde_json::from_value(json).unwrap();

        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1677652288);
        assert_eq!(response.model, "llama");
        assert_eq!(response.system_fingerprint, "fp_44709d6fcb");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(
            response.choices[0].message,
            Message::Assistant {
                content: Some(MessageContent::Text(
                    "Hello, how can I help you today?".to_string()
                )),
                name: None,
                refusal: None,
                tool_calls: vec![],
            }
        );
        assert_eq!(response.choices[0].finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.prompt_tokens, 9);
        assert_eq!(response.usage.completion_tokens, 12);
        assert_eq!(response.usage.total_tokens, 21);
    }

    #[test]
    fn test_deserialize_choice() {
        let json = json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, how can I help you today?"
            },
            "finish_reason": "stop"
        });

        let choice: Choice = serde_json::from_value(json).unwrap();

        assert_eq!(choice.index, 0);
        assert!(matches!(choice.message, Message::Assistant { .. }));
        assert!(matches!(choice.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn test_deserialize_finish_reason() {
        assert_eq!(
            serde_json::from_str::<FinishReason>("\"stop\"").unwrap(),
            FinishReason::Stop
        );
        assert_eq!(
            serde_json::from_str::<FinishReason>("\"length\"").unwrap(),
            FinishReason::Length
        );
        assert!(serde_json::from_str::<FinishReason>("\"content_filter\"").is_err());
    }

    #[test]
    fn a_completed_response_is_one_assistant_choice_with_its_finish_and_usage() {
        let response = ChatCompletionResponse::completed(
            identity(1_700_000_000),
            "Hello!".into(),
            FinishReason::Length,
            Usage::new(9, 12),
        );
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["created"], 1_700_000_000);
        assert_eq!(json["choices"][0]["index"], 0);
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert_eq!(json["usage"]["prompt_tokens"], 9);
        assert_eq!(json["usage"]["completion_tokens"], 12);
        assert_eq!(json["usage"]["total_tokens"], 21);
        assert_eq!(
            serde_json::from_value::<ChatCompletionResponse>(json).unwrap(),
            response
        );
    }

    #[test]
    fn a_text_chunk_carries_the_text_alone_and_the_last_chunk_the_finish_and_usage() {
        let text = ChatCompletionChunk::text(&identity(5), "Hel".into());
        let json = serde_json::to_value(&text).unwrap();
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(json["choices"][0]["delta"]["content"], "Hel");
        assert!(json["choices"][0].get("finish_reason").is_none());

        let last =
            ChatCompletionChunk::finished(&identity(5), FinishReason::Stop, Usage::new(3, 2));
        let json = serde_json::to_value(&last).unwrap();
        assert!(json["choices"][0]["delta"].get("content").is_none());
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    #[test]
    fn test_deserialize_usage() {
        let json = json!({
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        });

        let usage: Usage = serde_json::from_value(json).unwrap();

        assert_eq!(usage.prompt_tokens, 9);
        assert_eq!(usage.completion_tokens, 12);
        assert_eq!(usage.total_tokens, 21);
    }

    #[test]
    fn test_deserialize_choice_with_logprobs() {
        let json = json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, how can I help you today?"
            },
            "logprobs": {
                "token_logprobs": [-0.5, -0.2, -0.3],
                "top_logprobs": [
                    {"Hello": -0.5, "Hi": -0.7},
                    {"how": -0.2, "what": -0.4},
                    {"can": -0.3, "may": -0.5}
                ]
            },
            "finish_reason": "stop"
        });

        let choice: Choice = serde_json::from_value(json).unwrap();

        assert_eq!(choice.index, 0);
        assert!(matches!(choice.message, Message::Assistant { .. }));
        assert!(choice.logprobs.is_some());
        assert_eq!(
            choice.logprobs.unwrap(),
            json!({
                "token_logprobs": [-0.5, -0.2, -0.3],
                "top_logprobs": [
                    {"Hello": -0.5, "Hi": -0.7},
                    {"how": -0.2, "what": -0.4},
                    {"can": -0.3, "may": -0.5}
                ]
            })
        );
        assert!(matches!(choice.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn test_deserialize_delta() {
        let json = json!({
            "role": "assistant",
            "content": "Hello, how can I help you today?",
            "tool_calls": [],
            "refusal": "refusal"
        });

        let delta: Delta = serde_json::from_value(json).unwrap();

        assert_eq!(delta.role, "assistant");
        assert_eq!(
            delta.content,
            Some("Hello, how can I help you today?".to_string())
        );
        assert_eq!(delta.tool_calls, vec![]);
        assert_eq!(delta.refusal, Some("refusal".to_string()));
    }

    #[test]
    fn test_deserialize_stream_choice() {
        let json = json!({
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "Hello, how can I help you today?"
            }
        });

        let choice: StreamChoice = serde_json::from_value(json).unwrap();

        assert_eq!(choice.index, 0);
        assert_eq!(choice.delta.role, "assistant");
        assert_eq!(
            choice.delta.content,
            Some("Hello, how can I help you today?".to_string())
        );
    }

    #[test]
    fn test_deserialize_chat_completion_chunk() {
        let json = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello, how can I help you today?"
                }
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        });

        let chunk: ChatCompletionChunk = serde_json::from_value(json).unwrap();

        assert_eq!(chunk.id, "chatcmpl-123");
        assert_eq!(chunk.object, "chat.completion");
        assert_eq!(chunk.created, 1677652288);
        assert_eq!(chunk.model, "llama");
        assert_eq!(chunk.system_fingerprint, "fp_44709d6fcb");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(
            chunk.choices[0].delta,
            Delta {
                role: "assistant".to_string(),
                content: Some("Hello, how can I help you today?".to_string()),
                tool_calls: vec![],
                refusal: None,
            }
        );
        assert_eq!(chunk.usage.prompt_tokens, 9);
        assert_eq!(chunk.usage.completion_tokens, 12);
        assert_eq!(chunk.usage.total_tokens, 21);
    }
}
