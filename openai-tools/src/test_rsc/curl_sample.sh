#!/bin/bash
set -eu

ENV_FILE="../../../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "$ENV_FILE file not found!"
    exit 1
fi

# Source the .env file, handling spaces around =
set -a  # Enable auto-export
source "$ENV_FILE"
set +a  # Disable auto-export

echo "=== Sending request to OpenAI API ==="
echo

# Execute curl and capture response
echo "Making API request..."
response=$(curl -s --max-time 30 https://api.openai.com/v1/responses \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "この画像に何が写っていますか？"
                    },
                    {
                        "type": "input_image",
                        "image_url": "https://images.ctfassets.net/kftzwdyauwt9/1cFVP33AOU26mMJmCGDo1S/0029938b700b84cd7caed52124ed508d/OAI_BrandPage_11.png"
                    }
                ]
            }
        ]
    }')

echo "API request completed."

echo "=== Raw JSON Response ==="
# Use jq to pretty-print JSON if available, otherwise just echo
if command -v jq >/dev/null 2>&1; then
    echo "$response" | jq .
else
    echo "$response"
fi
echo

echo "=== Decoded Text Content ==="

# Function to decode Unicode escape sequences
decode_unicode() {
    local text="$1"
    # Try different approaches for Unicode decoding
    
    # Try with perl if available (most systems have it)
    if command -v perl >/dev/null 2>&1; then
        echo "$text" | perl -CS -pe 's/\\u([0-9a-fA-F]{4})/chr(hex($1))/eg' 2>/dev/null || echo "$text"
    # Try with node.js if available
    elif command -v node >/dev/null 2>&1; then
        echo "$text" | node -e "
            let input = '';
            process.stdin.setEncoding('utf8');
            process.stdin.on('data', (chunk) => input += chunk);
            process.stdin.on('end', () => {
                try {
                    console.log(JSON.parse('\"' + input.trim() + '\"'));
                } catch (e) {
                    console.log(input.trim());
                }
            });
        " 2>/dev/null || echo "$text"
    # Fallback to raw text
    else
        echo "$text"
    fi
}

# Extract and process response using shell tools
if command -v jq >/dev/null 2>&1; then
    # Check for error response
    error_message=$(echo "$response" | jq -r '.error.message // empty' 2>/dev/null)
    if [ -n "$error_message" ]; then
        echo "❌ API Error:"
        echo "   Message: $error_message"
        echo "   Type: $(echo "$response" | jq -r '.error.type // "Unknown type"')"
        error_code=$(echo "$response" | jq -r '.error.code // empty' 2>/dev/null)
        if [ -n "$error_code" ]; then
            echo "   Code: $error_code"
        fi
        exit 1
    fi
    
    # Extract text content from successful response
    output_exists=$(echo "$response" | jq -r '.output // empty' 2>/dev/null)
    if [ -n "$output_exists" ]; then
        echo "$response" | jq -r '.output[] | select(.type_name == "message") | .content[]? | select(.text) | .text' 2>/dev/null | while IFS= read -r text_content; do
            if [ -n "$text_content" ]; then
                echo "📝 Response Text:"
                # Decode Unicode escape sequences
                decoded_text=$(decode_unicode "$text_content")
                echo "$decoded_text"
                echo
            fi
        done
    else
        echo "⚠️  No output found in response"
    fi
else
    # Fallback without jq - simple grep/sed approach
    if echo "$response" | grep -q '"error"'; then
        echo "❌ API Error detected in response"
        echo "Raw response: $response"
        exit 1
    fi
    
    # Simple extraction without jq
    if echo "$response" | grep -q '"output"'; then
        # Extract text content using sed/grep (basic approach)
        text_content=$(echo "$response" | sed -n 's/.*"text":"\([^"]*\)".*/\1/p' | head -1)
        if [ -n "$text_content" ]; then
            echo "📝 Response Text:"
            decoded_text=$(decode_unicode "$text_content")
            echo "$decoded_text"
            echo
        fi
    else
        echo "⚠️  No output found in response"
    fi
fi