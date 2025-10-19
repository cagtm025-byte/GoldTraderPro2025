<#
deploy_azure.ps1
Usage example:
.\deploy_azure.ps1 -RG "GoldTraderPro-RG" -LOCATION "CentralIndia" -PLAN "GoldTraderPro-Plan" -APPNAME "goldtraderpro1234" -ZIP_PATH ".\GoldTraderPro2025.zip"
#>

param(
    [string]$RG = "goldtraderpro",
    [string]$LOCATION = "Central US",
    [string]$PLAN = "ASP-goldtraderpro-8ab2 (B1: 1)",
    [string]$APPNAME = "goldtraderpro",
    [string]$ZIP_PATH = "D:\Software\GoldTraderPro\GoldTraderPro2025.zip"
)

Write-Output "Resource group: $RG"
Write-Output "Location: $LOCATION"
Write-Output "App Service plan: $PLAN"
Write-Output "App name: $APPNAME"
Write-Output "Zip path: $ZIP_PATH"

# 1) Create resource group
az group create --name $RG --location $LOCATION

# 2) Create Linux App Service plan
az appservice plan create --name $PLAN --resource-group $RG --sku B1 --is-linux

# 3) Create the webapp using Python 3.11 runtime
az webapp create --resource-group $RG --plan $PLAN --name $APPNAME --runtime "PYTHON:3.11"

# 4) Set the port Azure will route to
az webapp config appsettings set --resource-group $RG --name $APPNAME --settings WEBSITES_PORT=8501

# 5) Configure startup file (startup.sh must exist in zip at root)
az webapp config set --resource-group $RG --name $APPNAME --startup-file "startup.sh"

# 6) Deploy via zip
az webapp deployment source config-zip --resource-group $RG --name $APPNAME --src $ZIP_PATH

Write-Output "Deployment done. Tail logs with:"
Write-Output "az webapp log tail --resource-group $RG --name $APPNAME"
