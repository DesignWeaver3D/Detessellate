# Update VarSet labels :
# 0. Oops, you want to change a variable name in VarSet :(
# 1. Execute VarSetUpdate macro from the Macro > Macros... dialog.
# 2. Use the dropdowns to select the VarSet and Poperty (variable) to be updated.
# 3. Modify the name or other attributes as necessary in the respective input fields.
# 4. Click Update button.
# 5. Voila !

__Name__ = "VarSet Update"
__Comment__ = "This macro updates the name or other attribute of a VarSet property (variable) via backup and recreation of the property."
__Author__ = "Mathias L., NSUBB"
__Version__ = "0.3.14"
__Date__ = "2025-04-24"
__License__ = "GNU GPL v3.0"
__Web__ = "https://github.com/NSUBB/VarSet-Update"
__Wiki__ = " "
__Icon__ = "VarSetUpdate.svg"
__IconW__ = " "
__Help__ = "After launch, select the VarSet, then the Property from the dropdowns, change attributes, click Update. Repeat for additional changes. Close when finished."
__Status__ = "Beta"
__Requires__ = "freecad 1.0"
__Communication__ = "https://github.com/NSUBB/VarSet-Update/issues"
__Files__ = "VarSetUpdate_v0.3.14.FCMacro, VarSetUpdate.svg"

# v0.1 3/26/2025 Original version by Mathias L. posted to https://github.com/FreeCAD/FreeCAD/issues/16222#issuecomment-2754714033
# v0.3.14 4/24/2025 Modified by NSUBB (FreeCAD Forum user DesignWeaver) to add many additional features.

import FreeCAD
from PySide import QtGui

class UpdateVarSetDialog(QtGui.QDialog):
    def __init__(self):
        super(UpdateVarSetDialog, self).__init__()

        self.setWindowTitle("Update a variable in VarSet")

        # Labels and input widgets
        self.search_name_label = QtGui.QLabel("Select VarSet:")
        self.search_name_input = QtGui.QComboBox()
        self.varset_label_display = QtGui.QLabel("")  # Label for VarSet
        self.varset_label_display.setAlignment(QtGui.Qt.AlignCenter)
        self.varset_label_display.setStyleSheet("color: #EE5F00; font-style: italic;") # Orange color for VarSet display name (Label)
        font = self.varset_label_display.font()
        font.setItalic(True)
        self.varset_label_display.setFont(font)

        self.old_name_label = QtGui.QLabel("Select Variable to Modify:")
        self.old_name_input = QtGui.QComboBox()
        
        # UserString horizontal layout
        self.user_string_label = QtGui.QLabel("Current Value:")
        self.user_string_label.setStyleSheet("font-weight: bold;")
        self.user_string_label.setAlignment(QtGui.Qt.AlignLeft)  # Align left within the horizontal layout

        self.user_string_value = QtGui.QLabel("")  # Value label for UserString
        #self.user_string_value.setStyleSheet("color: white;")
        self.user_string_value.setAlignment(QtGui.Qt.AlignRight)  # Align right within the horizontal layout

        self.user_string_layout = QtGui.QHBoxLayout()  # Create a horizontal layout
        self.user_string_layout.addWidget(self.user_string_label)
        self.user_string_layout.addWidget(self.user_string_value)

        # ExpressionEngine horizontal layout
        self.expression_engine_label = QtGui.QLabel("Current Expression:")
        self.expression_engine_label.setStyleSheet("font-weight: bold;")
        self.expression_engine_label.setAlignment(QtGui.Qt.AlignLeft)

        self.expression_engine_value = QtGui.QLabel("")  # Value label for ExpressionEngine
        self.expression_engine_value.setStyleSheet("color: #4AA5FF;") # Blue text color for expression to match appearance with FreeCAD GUI
        self.expression_engine_value.setAlignment(QtGui.Qt.AlignRight)

        self.expression_engine_layout = QtGui.QHBoxLayout()  # Create a horizontal layout
        self.expression_engine_layout.addWidget(self.expression_engine_label)
        self.expression_engine_layout.addWidget(self.expression_engine_value)


        self.new_name_label = QtGui.QLabel("New name:")
        self.new_name_input = QtGui.QLineEdit()

        self.property_type_label = QtGui.QLabel("Property Type:")
        self.property_type_input = QtGui.QComboBox()

        self.tooltip_label = QtGui.QLabel("New Tool Tip:")
        self.tooltip_input = QtGui.QLineEdit()

        self.group_name_label = QtGui.QLabel("Destination Group:")
        self.group_name_input = QtGui.QLineEdit()
        self.group_name_input.setText("Base")

        self.update_button = QtGui.QPushButton("Update")
        self.cancel_button = QtGui.QPushButton("Close")
        self.results_text = QtGui.QTextEdit("Warning!\nReplacing a property name may break existing relationships.\n")
        self.results_text.setReadOnly(True)

        # Layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.search_name_label)
        layout.addWidget(self.search_name_input)
        layout.addWidget(self.varset_label_display)
        layout.addWidget(self.old_name_label)
        layout.addWidget(self.old_name_input)

        # Insert the user_string_layout below old_name_input
        layout.addLayout(self.user_string_layout)

        # Insert the expression_engine_layout below user_string_layout
        layout.addLayout(self.expression_engine_layout)

        layout.addWidget(self.new_name_label)
        layout.addWidget(self.new_name_input)
        layout.addWidget(self.property_type_label)
        layout.addWidget(self.property_type_input)
        layout.addWidget(self.tooltip_label)
        layout.addWidget(self.tooltip_input)
        layout.addWidget(self.group_name_label)
        layout.addWidget(self.group_name_input)
        layout.addWidget(self.update_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.results_text)
        
        # Adjust layout settings
        layout.setSpacing(5)  # Reduce vertical spacing
        layout.setContentsMargins(10, 5, 10, 5)  # Set margins: (left, top, right, bottom)

        self.setLayout(layout)
        
        # Populate initial dropdowns
        self.populate_varset_dropdown()
        # Trigger initial label updates after old_name_input is populated
        if self.old_name_input.count() > 0:  # Ensure the dropdown has items
            self.old_name_input.setCurrentIndex(0)  # Select the first item
            self.on_property_selection_changed()  # Trigger label updates

        # Event connections
        self.search_name_input.currentIndexChanged.connect(self.update_property_dropdown)
        self.old_name_input.currentIndexChanged.connect(self.on_property_selection_changed)  # Trigger updates only when ready
        self.search_name_input.currentIndexChanged.connect(self.update_label_display)
        self.update_button.clicked.connect(self.update_variable)
        self.cancel_button.clicked.connect(self.close)

    def populate_varset_dropdown(self):
        """Populate the search_name_input dropdown with VarSet objects."""
        self.search_name_input.clear()
        doc = FreeCAD.ActiveDocument
        if doc is not None:
            for obj in doc.Objects:
                if obj.Name.startswith("VarSet"):
                    self.search_name_input.addItem(obj.Name)
        if self.search_name_input.count() > 0:
            self.search_name_input.setCurrentIndex(0)
            self.update_property_dropdown()
            self.update_label_display()

    def update_label_display(self):
        """Update the label to show the Label property of the selected VarSet object."""
        selected_varset_name = self.search_name_input.currentText()
        if not selected_varset_name:
            self.varset_label_display.setText("")
            return

        doc = FreeCAD.ActiveDocument
        if doc is None:
            self.varset_label_display.setText("")
            return

        varset = next((obj for obj in doc.Objects if obj.Name == selected_varset_name), None)
        if varset and hasattr(varset, "Label"):
            self.varset_label_display.setText(f"Label: {varset.Label}")
        else:
            self.varset_label_display.setText("")

    def update_property_dropdown(self):
        """Update the old_name_input dropdown with properties of the selected VarSet object."""
        self.old_name_input.clear()
        self.property_type_input.clear()
        selected_varset_name = self.search_name_input.currentText()

        if not selected_varset_name:
            return

        doc = FreeCAD.ActiveDocument
        if doc is None:
            return

        varset = next((obj for obj in doc.Objects if obj.Name == selected_varset_name), None)

        if varset:
            excluded_properties = {"ExpressionEngine", "Label", "Label2", "Visibility"}
            properties_list = [prop for prop in varset.PropertiesList if prop not in excluded_properties]
            self.old_name_input.addItems(properties_list)

            if properties_list:
                current_property = properties_list[0]
                self.old_name_input.setCurrentIndex(0)
                self.populate_property_type(varset.getTypeIdOfProperty(current_property))

        self.old_name_input.currentIndexChanged.connect(self.on_property_selection_changed)

    def on_property_selection_changed(self):
        """Update fields when the selected property changes."""
        selected_varset_name = self.search_name_input.currentText()  # Currently selected VarSet
        selected_property = self.old_name_input.currentText()  # Currently selected property within VarSet

        if not selected_varset_name or not selected_property:
            return

        doc = FreeCAD.ActiveDocument
        if doc is None:
            return

        # Fetch the selected VarSet object
        varset = next((obj for obj in doc.Objects if obj.Name == selected_varset_name), None)
        if varset:
            self.populate_property_type(varset.getTypeIdOfProperty(selected_property))
            self.update_prepopulated_fields(varset, selected_property)

            # Update UserString
            try:
                user_string = getattr(varset, selected_property, None)  # Access the property directly
                if hasattr(user_string, "UserString"):
                    self.user_string_value.setText(user_string.UserString if user_string.UserString else "None")
                else:
                    self.user_string_value.setText("None")
            except AttributeError:
                self.user_string_value.setText("None")

            # Update ExpressionEngine for the selected property
            try:
                expression_engine = getattr(varset, "ExpressionEngine", None)  # Access ExpressionEngine from the VarSet
                if expression_engine:
                    # Filter expressions related to the selected property
                    expression_items = [
                        f"{name} = {expression}" for name, expression in expression_engine if name == selected_property
                    ]
                    self.expression_engine_value.setText("\n".join(expression_items) if expression_items else "No Expressions Found")
                else:
                    self.expression_engine_value.setText("No Expressions Found")  # Handle empty ExpressionEngine
            except AttributeError:
                self.expression_engine_value.setText("Error Accessing ExpressionEngine")  # Handle any other error

 
    def populate_property_type(self, current_type):
        """Populate the Property Type dropdown with predefined options and preselect the current type."""
        self.property_type_input.clear()

        # List of possible types
        possible_types = [
            "App::PropertyBool",
            "App::PropertyAngle",
            "App::PropertyFloat",
            "App::PropertyDistance",
            "App::PropertyLength",
            "App::PropertyInteger",
            "App::PropertyString"
        ]

        self.property_type_input.addItems(possible_types)

        # Preselect the current type if available
        index = self.property_type_input.findText(current_type)
        if index != -1:
            self.property_type_input.setCurrentIndex(index)

    def update_prepopulated_fields(self, varset, property_name):
        """Prepopulate the fields with the current property's values."""
        # Prepopulate new_name_input with the current property name
        self.new_name_input.setText(property_name)

        # Prepopulate tooltip_input with the tooltip value (default to blank if no tooltip exists)
        try:
            tooltip = varset.getDocumentationOfProperty(property_name)
            self.tooltip_input.setText(tooltip if tooltip else "")
        except AttributeError:
            self.tooltip_input.setText("")  # Default to empty if tooltip is unavailable

        # Prepopulate group_name_input with the property's group name
        try:
            group_name = varset.getGroupOfProperty(property_name)
            self.group_name_input.setText(group_name if group_name else "Base")
        except AttributeError:
            self.group_name_input.setText("Base")  # Default to "Base" if group name is missing

    def show_conversion_popup(self, old_value, target_type):
        """Show a popup to prompt the user for a new value during conversion."""
        popup = QtGui.QInputDialog(self)
        popup.setWindowTitle("Unit Mismatch")
        popup.setLabelText(f"Converting to {target_type}\nOld Value: {old_value}\n\nEnter New Value:")
        popup.setTextValue(str(old_value))  # Prepopulate with the old value as a suggestion

        # Show popup and get user input
        if popup.exec() == QtGui.QDialog.Accepted:
            user_input = popup.textValue()
            try:
                # Validate the input based on the target type
                if target_type == "App::PropertyInteger":
                    return int(float(user_input))  # Allow float input but truncate/round to int
                elif target_type == "App::PropertyFloat":
                    return float(user_input)
                elif target_type == "App::PropertyAngle":
                    return float(user_input)  # Assume angle is in degrees
                elif target_type in ["App::PropertyString"]:
                    return str(user_input)
                elif target_type == "App::PropertyBool":
                    # Handle Bool as 1 (True) or 0 (False)
                    lower_input = user_input.strip().lower()
                    if lower_input in ["true", "1"]:
                        return True
                    elif lower_input in ["false", "0"]:
                        return False
                    else:
                        raise ValueError("Invalid Bool input. Enter 'True' or 'False'.")
                elif target_type in ["App::PropertyLength", "App::PropertyDistance"]:
                    return float(user_input)  # Length and distance are treated as floats
                else:
                    raise ValueError(f"Unsupported target type: {target_type}")
            except ValueError as e:
                QtGui.QMessageBox.warning(self, "Invalid Input", f"Error: {e}\nPlease try again.")
                return None  # Return None to indicate invalid input
        else:
            return None  # Return None if the user cancels the popup

    # The update_all_expressions method should now replace update_all_varset_expressions:
    def update_all_expressions(self, old_name, new_name):
        """Update ExpressionEngine entries for all objects to replace old_name with new_name."""
        doc = FreeCAD.ActiveDocument
        if doc is None:
            QtGui.QMessageBox.warning(self, "Error", "No active document found.")
            return

        updated_count = 0
        for obj in doc.Objects:
            expression_engine = getattr(obj, "ExpressionEngine", None)
            if expression_engine:
                for path, expression in expression_engine:
                    if old_name in expression:
                        updated_expression = expression.replace(old_name, new_name)
                        try:
                            obj.setExpression(path, updated_expression)
                            updated_count += 1
                            self.results_text.append(f"Updated expression for '{path}' in '{obj.Name}' to reference '{new_name}'.\n")
                        except Exception as e:
                            self.results_text.append(f"Failed to update expression for '{path}' in '{obj.Name}': {e}\n")

        doc.recompute()
        self.results_text.append(f"Updated {updated_count} expression(s) across all objects.\n")
    
    def update_variable(self):
        """Update the selected variable with the provided new name, type, tooltip, and group."""
        search_name = self.search_name_input.currentText()
        old_name = self.old_name_input.currentText()
        new_name = self.new_name_input.text().strip()

        if not search_name or not old_name or not new_name:
            QtGui.QMessageBox.warning(self, "Error", "Please enter all required fields.")
            return

        doc = FreeCAD.ActiveDocument
        if doc is None:
            QtGui.QMessageBox.warning(self, "Error", "No active document found.")
            return

        self.results_text.clear()
        self.results_text.append(f"Processing VarSet: '{search_name}'...\n")

        transaction_name = f"Update Variable: {old_name} to {new_name}"
        doc.openTransaction(transaction_name)

        try:
            varset = next((obj for obj in doc.Objects if obj.Name == search_name), None)

            if varset:
                if old_name in varset.PropertiesList:
                    selected_type = self.property_type_input.currentText()
                    old_value = getattr(varset, old_name, None)
                    group_name = self.group_name_input.text().strip() or "Base"

                    # Backup and clear expressions across the entire document
                    backed_up_expressions = []
                    for obj in doc.Objects:
                        expression_engine = getattr(obj, "ExpressionEngine", None)
                        if expression_engine:
                            for path, expression in expression_engine:
                                # Check if this expression references the old property
                                if old_name in path or old_name in expression:
                                    backed_up_expressions.append((obj, path, expression))
                                    obj.setExpression(path, None)  # Temporarily clear the expression
                                    self.results_text.append(f"Cleared expression for '{path}' in '{obj.Name}': '{expression}'.\n")

                    # Remove the old property
                    varset.removeProperty(old_name)
                    self.results_text.append(f"Removed property '{old_name}'.\n")

                    # Recreate the property
                    varset.addProperty(selected_type, new_name, group_name, self.tooltip_input.text())
                    setattr(varset, new_name, old_value)  # Assign the old value to the new property
                    self.results_text.append(f"Created property '{new_name}' with type '{selected_type}' in group '{group_name}' and value '{old_value}'.\n")

                    # Recompute the document to ensure the new property is recognized
                    doc.recompute()

                    # Restore expressions across the entire document
                    for obj, path, expression in backed_up_expressions:
                        updated_path = path.replace(old_name, new_name)  # Update path to reflect new property name
                        updated_expression = expression.replace(old_name, new_name)  # Update formula to reflect new property name
                        try:
                            obj.setExpression(updated_path, updated_expression)  # Restore updated expression
                            self.results_text.append(f"Restored expression for '{updated_path}' in '{obj.Name}': '{updated_expression}'.\n")
                        except Exception as e:
                            self.results_text.append(f"Failed to restore expression for '{updated_path}' in '{obj.Name}': {e}\n")

                    # Final recompute to ensure all expressions are resolved
                    doc.recompute()
                else:
                    self.results_text.append(f"Property '{old_name}' not found in VarSet '{search_name}'.\n")
            else:
                self.results_text.append(f"VarSet object '{search_name}' not found.\n")

            doc.commitTransaction()
            self.results_text.append("Update completed successfully!\n")

        except Exception as e:
            doc.abortTransaction()
            self.results_text.append("Transaction aborted. Use Undo to restore the previous state.")
            error_message = f"Error: {e}"
            self.results_text.append(error_message)
            try:
                QtGui.QMessageBox.warning(self, "Error", error_message)
            except Exception as inner_e:
                self.results_text.append(f"Failed to show error dialog: {inner_e}")


# Run the dialog
app = QtGui.QApplication.instance()
if not app:
    app = QtGui.QApplication([])

dialog = UpdateVarSetDialog()
dialog.exec()
