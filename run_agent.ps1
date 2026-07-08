param(
    [ValidateSet("menu", "sample", "live", "live-vision", "watch", "list", "debug", "debug-shop", "debug-trinket", "debug-triple", "desktop")]
    [string]$Mode = "menu",
    [string]$WindowTitle = "炉石传说",
    [string]$Profile = "examples\crop_profiles\doc_game_image3.json"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

function Test-Config {
    $envFile = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path -LiteralPath $envFile)) {
        Write-Host ".env not found. Create it from .env.example and set LLM_API_KEY." -ForegroundColor Red
        return $false
    }

    $content = Get-Content -Raw -Encoding UTF8 -LiteralPath $envFile
    if ($content -match "replace-with-your-openai-api-key|replace-me") {
        Write-Host ".env still contains a placeholder LLM_API_KEY. Edit .env before using --use-llm." -ForegroundColor Yellow
    }
    return $true
}

function Invoke-Sample {
    if (-not (Test-Config)) { return }
    python -m battlegrounds_agent.cli --state "examples\sample_state.json" --use-llm
}

function Invoke-LiveOnce {
    if (-not (Test-Config)) { return }
    python -m battlegrounds_agent.live_agent `
        --screen `
        --detect-game-area `
        --phase "shop-buy" `
        --card-json "examples\data\cards.enriched.json" `
        --trinket-json "examples\data\trinkets.json" `
        --turn 6 `
        --available-tribes "野兽,机械,龙,元素,海盗" `
        --use-template-hud `
        --use-llm
}

function Invoke-LiveOnceWithVisionHud {
    if (-not (Test-Config)) { return }
    python -m battlegrounds_agent.live_agent `
        --screen `
        --detect-game-area `
        --phase "shop-buy" `
        --card-json "examples\data\cards.enriched.json" `
        --trinket-json "examples\data\trinkets.json" `
        --available-tribes "野兽,机械,龙,元素,海盗" `
        --use-vision-hud
}

function Invoke-LiveWatch {
    if (-not (Test-Config)) { return }
    python -m battlegrounds_agent.live_agent `
        --screen `
        --detect-game-area `
        --phase "shop-buy" `
        --card-json "examples\data\cards.enriched.json" `
        --trinket-json "examples\data\trinkets.json" `
        --turn 6 `
        --available-tribes "野兽,机械,龙,元素,海盗" `
        --use-template-hud `
        --use-llm `
        --watch `
        --interval 2
}

function Invoke-WindowList {
    python -m battlegrounds_agent.window_capture --title $WindowTitle --list
}

function Invoke-DesktopAssistant {
    if (-not (Test-Config)) { return }
    python -m battlegrounds_agent.desktop_assistant
}

function Invoke-VisionDebug {
    python -m battlegrounds_agent.vision_debugger `
        --screen `
        --detect-game-area `
        --phase "shop-buy" `
        --output-dir "work\vision_debug" `
        --top-k 3
}

function Invoke-TrinketDebug {
    python -m battlegrounds_agent.vision_debugger `
        --screen `
        --detect-game-area `
        --phase "trinket-fullscreen" `
        --output-dir "work\vision_debug_trinket" `
        --top-k 5
}

function Invoke-TripleDebug {
    python -m battlegrounds_agent.vision_debugger `
        --screen `
        --detect-game-area `
        --phase "triple-discover" `
        --output-dir "work\vision_debug_triple" `
        --top-k 5
}

if ($Mode -eq "menu") {
    Write-Host ""
    Write-Host "Hearthstone Battlegrounds Agent"
    Write-Host "Project: $ProjectRoot"
    Write-Host ""
    Write-Host "1. Run sample state with OpenAI"
    Write-Host "2. Capture full screen once, read HUD locally, and plan"
    Write-Host "3. Capture full screen once, read HUD with vision, and plan"
    Write-Host "4. Watch full screen, read HUD locally every 2 seconds"
    Write-Host "5. List matching windows"
    Write-Host "6. Debug shop-buy vision only"
    Write-Host "7. Debug trinket vision only"
    Write-Host "8. Debug triple-discover vision only"
    Write-Host "9. Desktop assistant window"
    Write-Host "10. Set window title keyword"
    Write-Host "11. Exit"
    Write-Host ""
    $choice = Read-Host "Choose"
    switch ($choice) {
        "1" { $Mode = "sample" }
        "2" { $Mode = "live" }
        "3" { $Mode = "live-vision" }
        "4" { $Mode = "watch" }
        "5" { $Mode = "list" }
        "6" { $Mode = "debug-shop" }
        "7" { $Mode = "debug-trinket" }
        "8" { $Mode = "debug-triple" }
        "9" { $Mode = "desktop" }
        "10" {
            $customTitle = Read-Host "Window title keyword"
            if ($customTitle) {
                $WindowTitle = $customTitle
                $Mode = "list"
            } else {
                exit 0
            }
        }
        default { exit 0 }
    }
}

switch ($Mode) {
    "sample" { Invoke-Sample }
    "live" { Invoke-LiveOnce }
    "live-vision" { Invoke-LiveOnceWithVisionHud }
    "watch" { Invoke-LiveWatch }
    "list" { Invoke-WindowList }
    "debug" { Invoke-VisionDebug }
    "debug-shop" { Invoke-VisionDebug }
    "debug-trinket" { Invoke-TrinketDebug }
    "debug-triple" { Invoke-TripleDebug }
    "desktop" { Invoke-DesktopAssistant }
}
