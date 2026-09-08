; Installer definition for Ultimate Vocal Remover.

#ifndef AppVersion
  #error AppVersion must be supplied by build script
#endif

#ifndef BuildFlavor
  #define BuildFlavor "cpu"
#endif

#define AppName "Ultimate Vocal Remover"
#define AppPublisher "Ultimate Vocal Remover, Inc."
#define AppURL "https://github.com/Anjok07/ultimatevocalremovergui"
#define AppExeName "UVR.exe"

[Setup]
AppId={{652AA21C-E084-435C-8ED9-4A29AC2731F1}

AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} version {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}

VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
VersionInfoVersion={#AppVersion}
VersionInfoProductVersion={#AppVersion}

DefaultDirName={localappdata}\Programs\Ultimate Vocal Remover
DefaultGroupName=Ultimate Vocal Remover

DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

OutputDir=.\build\installer
OutputBaseFilename=UVR_v{#AppVersion}_{#BuildFlavor}_setup

SetupIconFile=.\gui_data\img\GUI-Icon.ico
UninstallDisplayIcon={app}\{#AppExeName}

Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern

UsePreviousAppDir=yes

#ifdef UseDiskSpanning
DiskSpanning=yes
; For compability with FAT32
DiskSliceSize=4290000000
#endif

[Files]
Source: ".\build\pyinstaller\dist\UVR\*"; \
  DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; These writable runtime directories are present in the official Windows package.
Name: "{app}\ensemble_temps"
Name: "{app}\temp_sample_clips"
Name: "{app}\tmp"
Name: "{app}\gui_data\saved_ensembles"
Name: "{app}\gui_data\saved_settings"
Name: "{app}\models\Apollo_Models\model_data\model_alias_data"
Name: "{app}\models\MDX_Net_Models\model_data\model_alias_data"
Name: "{app}\models\VR_Models\model_data\model_alias_data"

[Tasks]
Name: "desktopicon"; \
  Description: "Create a desktop shortcut"; \
  GroupDescription: "Additional shortcuts:"; \
  Flags: unchecked

[Icons]
Name: "{autoprograms}\Ultimate Vocal Remover"; \
  Filename: "{app}\{#AppExeName}"; \
  WorkingDir: "{app}"

Name: "{autodesktop}\Ultimate Vocal Remover"; \
  Filename: "{app}\{#AppExeName}"; \
  WorkingDir: "{app}"; \
  Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; \
  Description: "Launch Ultimate Vocal Remover"; \
  WorkingDir: "{app}"; \
  Flags: nowait postinstall skipifsilent
