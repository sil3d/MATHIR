' ============================================================================
' MATHIR Daemon Hidden Launcher (VBScript)
' ----------------------------------------------------------------------------
' Runs auto_start.bat without showing a console window.
' Place this file in the Windows Startup folder:
'   C:\Users\So-i-learn-3D\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\mathir_daemon.vbs
' ============================================================================

Option Explicit

' Resolve the .bat path next to the opencode bin directory.
' We use a literal absolute path so the script works no matter where the user
' is when it runs (Startup folder executes from %windir%\system32).
Const BAT_PATH = "C:\Users\So-i-learn-3D\.config\opencode\bin\auto_start.bat"

Dim shell
Set shell = CreateObject("WScript.Shell")

' WindowStyle = 0  -> hide the window completely
' WaitUntilFinished = False -> return immediately, run in background
' (If WaitUntilFinished were True, VBScript would block on the .bat and would
' itself need to be kept alive; False is what the Startup folder expects.)
shell.Run """" & BAT_PATH & """", 0, False

Set shell = Nothing
