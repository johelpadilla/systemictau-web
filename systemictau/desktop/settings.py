import os
import json
import customtkinter as ctk
import tkinter.messagebox
from typing import Dict, Any

class AppSettings:
    def __init__(self):
        # Default Settings
        self.default_settings = {
            "window_size": 20,
            "n_permutations": 200,
            "significance_level": 0.05,
            "max_rows_before_downsampling": 80000,
            "map_bubble_max_size": 25,
            "enable_mi": False,
            "mi_bins": 5,
            "default_output_folder": os.path.expanduser("~")
        }
        
        # Determine config file path
        self.config_dir = os.path.expanduser("~/.systemictau")
        self.config_file = os.path.join(self.config_dir, "config.json")
        
        self.current_settings = self.default_settings.copy()
        self.load_settings()

    def load_settings(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved_settings = json.load(f)
                    # Update current settings with saved ones, keeping defaults for missing keys
                    for k, v in saved_settings.items():
                        if k in self.current_settings:
                            self.current_settings[k] = v
            except Exception as e:
                print(f"Failed to load settings: {e}")
        else:
            self.save_settings() # Create default file

    def save_settings(self):
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
            
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.current_settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def get(self, key: str, default=None):
        return self.current_settings.get(key, default)

    def set(self, key: str, value: Any):
        if key in self.current_settings:
            self.current_settings[key] = value
            self.save_settings()

    def restore_defaults(self):
        self.current_settings = self.default_settings.copy()
        self.save_settings()


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, settings_mgr: AppSettings):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("500x450")
        self.settings_mgr = settings_mgr
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Notebook (Tabs)
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        
        self.tab_general = self.tabview.add("General")
        self.tab_bigdata = self.tabview.add("Big Data")
        self.tab_map = self.tabview.add("Map")
        self.tab_experimental = self.tabview.add("Experimental")
        
        self.vars = {}
        
        self._build_general_tab()
        self._build_bigdata_tab()
        self._build_map_tab()
        self._build_experimental_tab()
        
        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(padx=20, pady=(0, 20), fill="x")
        
        ctk.CTkButton(btn_frame, text="Restore Defaults", fg_color="#888", hover_color="#666", 
                     command=self._restore_defaults).pack(side="left")
                     
        ctk.CTkButton(btn_frame, text="Cancel", fg_color="#d9534f", hover_color="#c9302c", 
                     command=self.destroy).pack(side="right", padx=(10,0))
                     
        ctk.CTkButton(btn_frame, text="Save Settings", fg_color="#5cb85c", hover_color="#4cae4c",
                     command=self._save_and_close).pack(side="right")

    def _add_setting_row(self, parent, label_text, key, val_type=str):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(frame, text=label_text, width=200, anchor="w").pack(side="left")
        
        current_val = self.settings_mgr.get(key)
        var = ctk.StringVar(value=str(current_val))
        self.vars[key] = (var, val_type)
        
        entry = ctk.CTkEntry(frame, textvariable=var, width=150)
        entry.pack(side="right")
        return frame

    def _add_checkbox_row(self, parent, label_text, key):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=10)
        
        current_val = self.settings_mgr.get(key)
        var = ctk.BooleanVar(value=bool(current_val))
        self.vars[key] = (var, bool)
        
        cb = ctk.CTkCheckBox(frame, text=label_text, variable=var)
        cb.pack(side="left")
        return frame

    def _build_general_tab(self):
        self._add_setting_row(self.tab_general, "Window Size (W):", "window_size", int)
        self._add_setting_row(self.tab_general, "Monte Carlo Permutations:", "n_permutations", int)
        self._add_setting_row(self.tab_general, "Significance Level (Alpha):", "significance_level", float)

    def _build_bigdata_tab(self):
        self._add_setting_row(self.tab_bigdata, "Max Rows Before Downsample:", "max_rows_before_downsampling", int)
        
        info = ctk.CTkLabel(self.tab_bigdata, text="* Used to prevent memory overload with huge datasets.", 
                            text_color="gray", font=("Arial", 11))
        info.pack(pady=10, anchor="w")

    def _build_map_tab(self):
        self._add_setting_row(self.tab_map, "Max Bubble Size:", "map_bubble_max_size", int)

    def _build_experimental_tab(self):
        self._add_checkbox_row(self.tab_experimental, "Enable Non-linear Mutual Information (MI)", "enable_mi")
        self._add_setting_row(self.tab_experimental, "MI Histogram Bins:", "mi_bins", int)
        
        info = ctk.CTkLabel(self.tab_experimental, text="* 5 bins avoid noisy estimates in small windows.\n* Warning: Enabling MI significantly increases computation time.", 
                            text_color="gray", font=("Arial", 11), justify="left")
        info.pack(pady=10, anchor="w")

    def _save_and_close(self):
        try:
            for key, (var, val_type) in self.vars.items():
                val = val_type(var.get())
                self.settings_mgr.current_settings[key] = val
                
            self.settings_mgr.save_settings()
            tkinter.messagebox.showinfo("Success", "Settings saved successfully.", parent=self)
            self.destroy()
        except ValueError as e:
            tkinter.messagebox.showerror("Validation Error", "Please ensure all fields contain valid numbers.", parent=self)
            
    def _restore_defaults(self):
        if tkinter.messagebox.askyesno("Confirm", "Are you sure you want to restore default settings?", parent=self):
            self.settings_mgr.restore_defaults()
            # Update UI
            for key, (var, _) in self.vars.items():
                var.set(str(self.settings_mgr.get(key)))
