# deploy_azure.ps1 (filled)
$RG = "GoldTrader Pro"
$LOCATION = "CentralIndia"
$PLAN = "GoldTrader Pro Plan"
$APPNAME = "Goldtraderpro"
$ZIP_PATH = ".\\GoldTrader.zip"
$PY_RUNTIME = "PYTHON:3.11"

Write-Output "Logging into Azure..."
az login | Out-Null

Write-Output "Creating resource group: $RG in $LOCATION"
az group create --name $RG --location $LOCATION | Out-Null

Write-Output "Creating App Service plan: $PLAN (Linux, SKU B1)"
az appservice plan create --name $PLAN --resource-group $RG --sku B1 --is-linux | Out-Null

Write-Output "Creating web app: $APPNAME with runtime $PY_RUNTIME"
az webapp create --resource-group $RG --plan $PLAN --name $APPNAME --runtime $PY_RUNTIME | Out-Null

Write-Output "Setting WEBSITES_PORT=8000"
az webapp config appsettings set --resource-group $RG --name $APPNAME --settings WEBSITES_PORT=8000 | Out-Null

Write-Output "Setting startup command to run Streamlit"
$STARTUP_CMD = 'python -m streamlit run Final/Final.py --server.port 8000 --server.address 0.0.0.0'
az webapp config set --resource-group $RG --name $APPNAME --startup-file $STARTUP_CMD | Out-Null

if (-Not (Test-Path $ZIP_PATH)) {
    Write-Error ("ZIP file not found at {0}. Place {1} next to this script or update ZIP_PATH." -f $ZIP_PATH, "GoldTrader.zip")
    exit 2
}

Write-Output "Deploying $ZIP_PATH to $APPNAME ..."
az webapp deployment source config-zip --resource-group $RG --name $APPNAME --src $ZIP_PATH | Out-Null

Write-Output "Deployment finished. Visit: https://$APPNAME.azurewebsites.net"