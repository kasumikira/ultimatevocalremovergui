; Installer definition for Ultimate Vocal Remover.

#ifndef AppVersion
  #error AppVersion must be supplied by build script
#endif

#define AppName "Ultimate Vocal Remover"
#define AppPublisher "Ultimate Vocal Remover, Inc."
#define AppExeName "UVR.exe"

[Setup]
AppId={{652AA21C-E084-435C-8ED9-4A29AC2731F1}

AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} version {#AppVersion}
AppPublisher={#AppPublisher}

VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
VersionInfoVersion={#AppVersion}
VersionInfoProductVersion={#AppVersion}

DefaultDirName={localappdata}\Programs\Ultimate Vocal Remover
DefaultGroupName=Ultimate Vocal Remover

DisableProgramGroupPage=yes
PrivilegesRequired=lowest

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

OutputDir=..\dist-installer
OutputBaseFilename=UVR_v{#AppVersion}_setup

SetupIconFile=..\gui_data\img\GUI-Icon.ico
UninstallDisplayIcon={app}\{#AppExeName}

Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

UsePreviousAppDir=yes

[Files]
Source: "..\dist\UVR\*"; \
  DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\tmp"

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