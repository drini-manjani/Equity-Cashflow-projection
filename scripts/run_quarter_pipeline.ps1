param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsList
)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..")
$ScriptPath = Join-Path $RootDir "scripts\run_quarter_pipeline.py"
python $ScriptPath @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
