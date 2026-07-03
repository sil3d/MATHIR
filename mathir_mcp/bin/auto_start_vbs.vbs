' ============================================================================
' MATHIR Daemon Hidden Launcher (VBScript)
' ----------------------------------------------------------------------------
' Runs auto_start.bat without showing a console window.
' Place this file in the Windows Startup folder:
'   %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\mathir_daemon.vbs
' ============================================================================

Option Explicit

' Resolve the .bat path next to the opencode bin directory.
' We use a literal absolute path so the script works no matter where the user
' is when it runs (Startup folder executes from %windir%\system32).
' Resolve dynamically from %USERPROFILE% so the script works for any user.
Dim shell2 : Set shell2 = CreateObject("WScript.Shell")
Dim BAT_PATH : BAT_PATH = shell2.ExpandEnvironmentStrings("%USERPROFILE%") & "\.config\MATHIR\mathir_mcp\bin\auto_start.bat"
Set shell2 = Nothing

Dim shell
Set shell = CreateObject("WScript.Shell")

' WindowStyle = 0  -> hide the window completely
' WaitUntilFinished = False -> return immediately, run in background
' (If WaitUntilFinished were True, VBScript would block on the .bat and would
' itself need to be kept alive; False is what the Startup folder expects.)
shell.Run """" & BAT_PATH & """", 0, False

Set shell = Nothing
