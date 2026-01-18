import customtkinter as ctk
import os
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from core.database import load_db, delete_user


# --- UI CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue") # Professional color theme

class FaceManagerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Smiling Face Admin | Database Manager")
        self.geometry("1000x700")
        
        # Data Cache
        self.all_names = []
        self.all_angles = []
        
        # Paths
        self.logo_path = "./assets/logo.png" 

        # --- MAIN LAYOUT ---
        # We use a main container to add padding around everything
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # 1. HEADER SECTION (Logo + Title + Stats)
        self.build_header()

        # 2. CONTROL BAR (Search + Filter + Refresh)
        self.build_controls()

        # 3. DATA TABLE (Headers + Scrollable List)
        self.build_table()

        # Initial Load
        self.full_reload_db()

    def build_header(self):
        """Creates the top dashboard header"""
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 15))

        # -- Left: Logo & Branding --
        branding_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        branding_frame.pack(side="left")

        # Logo Processing
        self.logo_image = self.load_circular_image(self.logo_path, size=(70, 70))
        if self.logo_image:
            lbl_logo = ctk.CTkLabel(branding_frame, text="", image=self.logo_image)
            lbl_logo.pack(side="left", padx=(0, 15))

        # Title Text
        text_frame = ctk.CTkFrame(branding_frame, fg_color="transparent")
        text_frame.pack(side="left")
        ctk.CTkLabel(text_frame, text="SMILING SECURITY", font=("Roboto Medium", 24), text_color="white").pack(anchor="w")
        ctk.CTkLabel(text_frame, text="Face Recognition Database Admin", font=("Segoe UI", 13), text_color="#9ca3af").pack(anchor="w")

        # -- Right: Stats Cards --
        stats_frame = ctk.CTkFrame(header_frame, fg_color="#2b2b2b", corner_radius=10)
        stats_frame.pack(side="right", ipady=5, ipadx=10)

        self.lbl_total_vectors = ctk.CTkLabel(stats_frame, text="Vectors: 0", font=("Roboto", 14, "bold"), text_color="#60a5fa")
        self.lbl_total_vectors.pack(side="left", padx=15)
        
        # Separator
        ctk.CTkFrame(stats_frame, width=2, height=20, fg_color="#444").pack(side="left")

        self.lbl_unique_users = ctk.CTkLabel(stats_frame, text="Users: 0", font=("Roboto", 14, "bold"), text_color="#4ade80")
        self.lbl_unique_users.pack(side="left", padx=15)

    def build_controls(self):
        """Creates the search bar and filter options"""
        control_frame = ctk.CTkFrame(self.main_container, fg_color="#2b2b2b", height=50, corner_radius=8)
        control_frame.pack(fill="x", pady=(0, 15))

        # Search Icon & Entry
        ctk.CTkLabel(control_frame, text="🔍", font=("Arial", 16)).pack(side="left", padx=(15, 5), pady=10)
        
        self.search_var = ctk.StringVar()
        self.search_var.trace("w", self.filter_data)
        
        self.entry_search = ctk.CTkEntry(control_frame, 
                                         placeholder_text="Search by name...", 
                                         width=300, 
                                         height=35,
                                         border_width=0,
                                         fg_color="#1a1a1a",
                                         textvariable=self.search_var)
        self.entry_search.pack(side="left", padx=5, pady=10)

        # Filter Dropdown
        ctk.CTkLabel(control_frame, text="Angle Filter:", font=("Segoe UI", 12, "bold"), text_color="gray").pack(side="left", padx=(20, 5))
        
        self.filter_angle_var = ctk.StringVar(value="All Angles")
        self.filter_combo = ctk.CTkOptionMenu(control_frame, 
                                              values=["All Angles", "straight", "left_side", "right_side"],
                                              command=lambda x: self.filter_data(), 
                                              variable=self.filter_angle_var,
                                              width=130,
                                              fg_color="#1a1a1a",
                                              button_color="#3b8ed0")
        self.filter_combo.pack(side="left", padx=5)

        # Refresh Button
        ctk.CTkButton(control_frame, text="↻ Refresh Data", 
                      width=120, 
                      fg_color="#374151", 
                      hover_color="#4b5563",
                      command=self.full_reload_db).pack(side="right", padx=15, pady=10)

    def build_table(self):
        """Creates the main data list"""
        # Table Header
        header_frame = ctk.CTkFrame(self.main_container, fg_color="#111827", height=40, corner_radius=6)
        header_frame.pack(fill="x", pady=(0, 5))
        
        self.setup_grid_columns(header_frame)

        ctk.CTkLabel(header_frame, text="#", font=("Arial", 12, "bold"), text_color="gray").grid(row=0, column=0, pady=10, padx=15, sticky="w")
        ctk.CTkLabel(header_frame, text="FULL NAME", font=("Arial", 12, "bold"), text_color="gray").grid(row=0, column=1, pady=10, sticky="w")
        ctk.CTkLabel(header_frame, text="FACE ANGLE", font=("Arial", 12, "bold"), text_color="gray").grid(row=0, column=2, pady=10, sticky="w")
        ctk.CTkLabel(header_frame, text="ACTIONS", font=("Arial", 12, "bold"), text_color="gray").grid(row=0, column=3, pady=10)

        # Scrollable List
        self.scroll_frame = ctk.CTkScrollableFrame(self.main_container, fg_color="#1f2937", corner_radius=6)
        self.scroll_frame.pack(fill="both", expand=True)
        self.setup_grid_columns(self.scroll_frame)

    def setup_grid_columns(self, frame):
        """Defines column weights for consistent alignment"""
        frame.grid_columnconfigure(0, weight=1) # Index
        frame.grid_columnconfigure(1, weight=4) # Name
        frame.grid_columnconfigure(2, weight=2) # Angle
        frame.grid_columnconfigure(3, weight=1) # Button

    # --- LOGIC FUNCTIONS ---

    def load_circular_image(self, image_path, size=(80, 80)):
        if not os.path.exists(image_path): return None
        try:
            img = Image.open(image_path).convert("RGBA")
            img = img.resize(size, Image.Resampling.LANCZOS)
            mask = Image.new("L", size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0) + size, fill=255)
            output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
            output.putalpha(mask)
            return ctk.CTkImage(light_image=output, dark_image=output, size=size)
        except Exception: return None

    def full_reload_db(self):
        try:
            vectors, names_dict = load_db()
            if not vectors or len(vectors) == 0:
                self.all_names = []
                self.all_angles = []
            else:
                self.all_names = names_dict.get('name', [])
                self.all_angles = names_dict.get('face_angle', [])
        except:
            self.all_names = []
            self.all_angles = []
        
        self.filter_data()

    def execute_delete(self, name):
        # English Warning Message
        msg = f"⚠️ WARNING: You are deleting by Name.\n\nAction will remove ALL records for user:\n👉 '{name}'\n\nAre you sure?"
        
        if messagebox.askyesno("Confirm Deletion", msg):
            try:
                delete_user(name) # Calls your core function
                self.full_reload_db()
                messagebox.showinfo("Success", f"Successfully deleted user: {name}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def filter_data(self, *args):
        # Clear current list
        for widget in self.scroll_frame.winfo_children(): widget.destroy()

        keyword = self.search_var.get().lower()
        angle_mode = self.filter_angle_var.get()
        
        display_count = 0
        
        for i in range(len(self.all_names)):
            name = self.all_names[i]
            angle = self.all_angles[i]

            # Logic: Name match AND (Angle match OR All)
            if (keyword in name.lower()) and ((angle_mode == "All Angles") or (angle_mode in angle)):
                self.create_row(i, name, angle, display_count)
                display_count += 1

        # Update Stats Cards
        unique_users = len(set(name.lower() for name in self.all_names))
        
        self.lbl_total_vectors.configure(text=f"Vectors: {display_count}")
        self.lbl_unique_users.configure(text=f"Users: {unique_users}")

    def create_row(self, index, name, angle, display_index):
        # Zebra Striping: Darker background for even rows
        bg_color = "transparent" if display_index % 2 == 0 else "#2d3748"
        
        row_frame = ctk.CTkFrame(self.scroll_frame, fg_color=bg_color, corner_radius=4, height=45)
        row_frame.grid(row=display_index, column=0, columnspan=4, sticky="ew", pady=1)
        self.setup_grid_columns(row_frame)

        # 1. Index
        ctk.CTkLabel(row_frame, text=f"{display_index + 1}", text_color="gray").grid(row=0, column=0, padx=15, sticky="w")
        
        # 2. Name (Bold)
        ctk.CTkLabel(row_frame, text=name, font=("Roboto", 13, "bold"), text_color="white").grid(row=0, column=1, sticky="w")

        # 3. Angle Badge
        # Color coding
        color = "#9ca3af" # Gray
        if angle == "straight": color = "#60a5fa" # Blue
        elif "left" in angle: color = "#facc15" # Yellow
        elif "right" in angle: color = "#f472b6" # Pink
        
        ctk.CTkLabel(row_frame, text=f"●  {angle}", text_color=color, font=("Segoe UI", 12)).grid(row=0, column=2, sticky="w")

        # 4. Delete Button
        btn = ctk.CTkButton(row_frame, text="Delete", width=80, height=26,
                            fg_color="#ef4444", hover_color="#b91c1c",
                            font=("Arial", 11, "bold"),
                            command=lambda n=name: self.execute_delete(n))
        btn.grid(row=0, column=3, pady=6)

if __name__ == "__main__":
    app = FaceManagerApp()
    app.mainloop()