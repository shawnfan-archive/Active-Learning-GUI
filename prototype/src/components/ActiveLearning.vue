<template>
  <v-app id="main">
    <v-container align-center justify-center>
      <!-- Toolbar -->
      <v-row>
        <v-toolbar dense prominent class="my-8">
          <v-app-bar-nav-icon></v-app-bar-nav-icon>

          <v-toolbar-title>An Active Learning Approach to Acute Stroke Detection</v-toolbar-title>

          <v-spacer></v-spacer>

          <v-toolbar-items>
            <v-btn text v-on:click="openDialog">
              Save and Retrain Model
              <v-icon>mdi-upload</v-icon>
            </v-btn>
            <v-btn text>
              Settings
              <v-icon>mdi-settings</v-icon>
            </v-btn>
          </v-toolbar-items>
        </v-toolbar>
      </v-row>

      <!-- User interface -->
      <v-row align="center" justify="center">
        <v-col cols="8">
          <v-container justify-center>
            <v-card class="pa-auto">
              <v-container>
                <v-row align="center" justify="center">
                  <!-- Buttons -->
                  <v-col cols="auto">
                    <!-- Paintbrush -->
                    <v-col>
                      <v-menu top offset-y>
                        <template v-slot:activator="{ on }">
                          <v-btn class="mx-2" fab dark color="red" v-on="on">
                            <v-icon dark>mdi-pencil</v-icon>
                          </v-btn>
                        </template>

                        <v-list rounded nav>
                          <v-list-item v-for="parameter in button_parameters" :key="parameter[0]">
                            <v-btn class="mx-1" fab dark color="orange" v-on:click="setTool('activate',parameter[0])">
                              <v-icon :size="parameter[1]">mdi-checkbox-blank-circle</v-icon>
                            </v-btn>
                          </v-list-item>

                        </v-list>
                      </v-menu>
                    </v-col>

                    <!-- Eraser -->
                    <v-col>
                      <v-menu bottom offset-y>
                        <template v-slot:activator="{ on }">
                          <v-btn class="mx-2" fab dark color="cyan" v-on="on">
                            <v-icon dark>mdi-eraser</v-icon>
                          </v-btn>
                        </template>

                        <v-list rounded nav>
                          <v-list-item v-for="parameter in button_parameters" :key="parameter[0]">
                            <v-btn class="mx-1" fab dark color="blue" v-on:click="setTool('deactivate',parameter[0])">
                              <v-icon :size="parameter[1]">mdi-checkbox-blank-circle</v-icon>
                            </v-btn>
                          </v-list-item>
                        </v-list>
                      </v-menu>
                    </v-col>

                  </v-col>

                  <!-- Canvases -->
                  <v-col cols="auto">
                    <v-row align="center" justify="center">
                      <h2>Image ID: {{ current_image.image_id }}</h2>

                      <v-divider class="mx-4" vertical></v-divider>

                      <h2>Disease: {{ current_image.disease }}</h2>
                    </v-row>

                    <v-row>
                      <div class="canvas">
                        <v-skeleton-loader
                          v-if="skeleton_loader"
                          :height="canvas_height"
                          :width="canvas_width"
                          type="image"
                        ></v-skeleton-loader>
                      </div>

                      <!--canvas for brain image-->
                      <div class="canvas">
                        <canvas ref="img" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
                      </div>
                      <!--canvas for activation map-->
                      <div class="canvas">
                        <canvas ref="map" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
                      </div>
                      <div class="canvas">
                        <canvas
                          ref="cursor"
                          v-bind:width="canvas_width"
                          v-bind:height="canvas_height"
                        ></canvas>
                      </div>
                      <!--canvas for corrections-->
                      <div id="draw">
                        <canvas
                          ref="draw"
                          v-bind:width="canvas_width"
                          v-bind:height="canvas_height"
                          v-on:mousemove="correctActivation"
                        ></canvas>
                      </div>
                    </v-row>
                  </v-col>
                </v-row>
              </v-container>
            </v-card>
          </v-container>
          <!-- </v-row> -->
        </v-col>

        <!-- Thumbnail list -->
        <v-col cols="2">
          <v-row justify="center">
            <v-card>
              <v-list class="overflow-y-auto" max-height="600">
                <v-list-item-group>
                  <v-list-item class="ma-4" v-for="image in images" :key="image.id">
                    <v-card class="pa-2" width="100" height="105" raised>
                      <v-img
                        class="my-0"
                        contain
                        :src="require(`../assets/${image.path}.jpeg`)"
                        weight="75"
                        height="75"
                        v-on:click="loadImage(image)"
                      ></v-img>
                      <v-card-subtitle class="pa-0">{{image.image_id}}</v-card-subtitle>
                    </v-card>
                  </v-list-item>
                </v-list-item-group>
              </v-list>
            </v-card>
          </v-row>
        </v-col>
      </v-row>

      <!-- Submission dialog -->
      <v-dialog v-model="dialog" persistent max-width="1000">
        <v-card>
          <v-card-title class="headline">Submit activaion maps and retrain model?</v-card-title>
          <v-card-text>Latest model: {{model_name}}</v-card-text>
          <v-card-actions>
            <v-btn color="red darken-1" text v-on:click="closeDialog">Cancel</v-btn>

            <v-spacer></v-spacer>

            <v-btn color="green darken-1" text v-on:click="onSubmit(true)">Train model from scratch</v-btn>
            <v-btn color="green darken-1" text v-on:click="onSubmit(false)">Retrain model</v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>

      <!-- Loading dialog -->
      <v-dialog v-model="loading" persistent width="800">
        <v-card color="primary" dark>
          <v-card-title>
            {{ loading_message }}
            <v-progress-linear indeterminate color="white"></v-progress-linear>
          </v-card-title>
        </v-card>
      </v-dialog>
    </v-container>
  </v-app>
</template>>

<script>
import axios from "axios";

export default {
  data() {
    return {
      current_image: {
        image_id: "",
        disease: "",
        path: ""
      },
      images: [
        // {
        //   image_id: string,
        //   disease: string,
        //   path: string
        // },
      ],
      activation_maps: [
        // {string (id): array(activation map)}
      ],
      tool: "deactivate",
      tool_started: false,
      tool_size: 10,
      button_parameters: [[20, 25], [10, 12.5], [5, 7.5]], 
      canvas_width: 600,
      canvas_height: 500,
      graphics: {
        // rgba
        activation_color: [255, 0, 0, 100],
        inactivation_color: [0, 0, 0, 0],
        paintbrush: "rgba(0, 0, 255, 255)",
        image_opacity: 1.0,
        map_opacity: 0.8
      },
      skeleton_loader: true,
      skeleton_thumbnail: [1, 2, 3, 4, 5],
      dialog: false,
      model_name: "",
      loading: false,
      check_number: 1,
      current_epoch: null,
      total_epochs: null,
      time_remaining: "",
      loading_message: "Initializing Training..."
    };
  },
  watch: {
    check_number: function() {
      if (this.loading) {
        // check training progress every 10 seconds
        setTimeout(this.updateTrainingProgress, 10000);
      } else {
        return null;
      }
    }
  },
  methods: {
    setTool: function(tool, tool_size) {
      this.tool = tool;
      this.tool_size = tool_size;
    },
    getActivationMap: function() {
      const path = "http://localhost:5000/active_learning";
      axios
        .get(path)
        .then(res => {
          this.images = res.data.images;
          this.model_name = res.data.latest_model;

          for (let image_id in res.data.activation_maps) {
            let activation_map = [];
            // convert binary array to rgba array
            for (
              let row_index = 0;
              row_index < this.canvas_height;
              row_index++
            ) {
              for (
                let col_index = 0;
                col_index < this.canvas_width;
                col_index++
              ) {
                if (
                  res.data.activation_maps[image_id][row_index][col_index] === 1
                ) {
                  activation_map.push.apply(
                    activation_map,
                    this.graphics.activation_color
                  );
                } else {
                  activation_map.push.apply(
                    activation_map,
                    this.graphics.inactivation_color
                  );
                }
              }
            }
            this.activation_maps[image_id] = activation_map;
          }

          this.skeleton_loader = false;

          this.current_image.image_id = "";

          // load first image in images
          this.loadImage(this.images[0]);
        })
        .catch(error => {
          console.error(error);
        });
    },
    loadImage: function(image) {
      // store activation map of old image if not first time loading
      if (this.current_image.image_id !== "") {
        this.storeActivationMap();
      }

      // update current image to new image
      this.current_image = image;

      // set image opacity
      let img_ctx = this.$refs.img.getContext("2d");
      img_ctx.globalAlpha = this.graphics.image_opacity;

      // load new image
      let img = new Image();
      img.src = require(`../assets/${this.current_image.path}.jpeg`);
      img.onload = () => {
        img_ctx.drawImage(img, 0, 0, this.canvas_width, this.canvas_height);
      };

      // load new activation map
      this.loadActivationMap();
    },
    loadCursor: function(x, y) {
      let ctx = this.$refs.cursor.getContext("2d");
      ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);
      ctx.fillStyle = "rgba(255,255,255,255)";
      ctx.beginPath();
      ctx.arc(x, y, this.tool_size / 2, 0, 2 * Math.PI);
      ctx.fill();
    },
    loadActivationMap: function() {
      // load activation map for current_image

      let map_ctx = this.$refs.map.getContext("2d");
      map_ctx.globalAlpha = this.graphics.map_opacity;
      let map_data = map_ctx.createImageData(
        this.canvas_width,
        this.canvas_height
      );

      let current_activation_map = this.activation_maps[
        this.current_image.image_id
      ];
      for (
        let pixel_index = 0;
        pixel_index < current_activation_map.length;
        pixel_index++
      ) {
        map_data.data[pixel_index] = current_activation_map[pixel_index];
      }

      map_ctx.putImageData(map_data, 0, 0);
    },
    correctActivation: function(event) {
      // load cursor
      this.loadCursor(event.offsetX, event.offsetY);

      // highlight incorrect activation
      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");
      ctx.strokeStyle = this.graphics.paintbrush;
      ctx.fillStyle = this.graphics.paintbrush;
      ctx.globalCompositeOperation = "source-over";
      ctx.lineJoin = "round";
      ctx.lineWidth = this.tool_size;
      // start drawing
      canvas.onmousedown = () => {
        // draw circle
        ctx.beginPath();
        ctx.arc(
          event.offsetX,
          event.offsetY,
          this.tool_size / 2,
          0,
          2 * Math.PI
        );
        ctx.fill();
        this.updateActivationMap();
        this.tool_started = true;
        ctx.beginPath();
        ctx.moveTo(event.offsetX, event.offsetY);
      };
      if (this.tool_started) {
        ctx.lineTo(event.offsetX, event.offsetY);
        ctx.stroke();
        this.updateActivationMap();
      }
      canvas.onmouseup = () => {
        // stop drawing
        if (this.tool_started) {
          this.tool_started = false;
        }
        // draw circle
        ctx.beginPath();
        ctx.arc(
          event.offsetX,
          event.offsetY,
          this.tool_size / 2,
          0,
          2 * Math.PI
        );
        ctx.fill();
        this.updateActivationMap();
      };
    },
    updateActivationMap: function() {
      let map_ctx = this.$refs.map.getContext("2d");
      let draw_ctx = this.$refs.draw.getContext("2d");
      let map_data = map_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );
      let correction_data = draw_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );

      let pixel_color = this.graphics.activation_color;
      if (this.tool === "deactivate") {
        pixel_color = this.graphics.inactivation_color;
      }

      for (let i = 0; i < correction_data.data.length; i += 4) {
        if (correction_data.data[i + 2] === 255) {
          map_data.data[i] = pixel_color[0];
          map_data.data[i + 1] = pixel_color[1];
          map_data.data[i + 2] = pixel_color[2];
          map_data.data[i + 3] = pixel_color[3];
        }
      }

      // draw updated activation map
      map_ctx.putImageData(map_data, 0, 0);

      // clear drawings
      let clear_data = map_ctx.createImageData(map_data);
      draw_ctx.putImageData(clear_data, 0, 0);
    },
    storeActivationMap: function() {
      // store activation map of current image

      let map_ctx = this.$refs.map.getContext("2d");

      let map_data = map_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );

      this.activation_maps[this.current_image.image_id] = map_data.data;
    },
    openDialog: function() {
      this.dialog = true;
    },
    closeDialog: function() {
      this.dialog = false;
    },
    saveData: function(payload) {
      const path = "http://localhost:5000/active_learning";
      axios
        .post(path, payload)
        .then(() => {
          this.loading = false;
          this.getActivationMap();
        })
        .catch(error => {
          console.log(error);
          this.getActivationMap();
        });

      setTimeout(this.updateTrainingProgress, 10000);
    },
    onSubmit: function(from_scratch) {
      this.storeActivationMap();
      this.loading_message = "Initializing Training..."; //Reset loading message
      
      // convert rgba array to binary array
      let corrected_activation_maps = {};
      for (let image_id in this.activation_maps) {
        let map_array = [];
        for (
          let pixel_index = 0;
          pixel_index < this.activation_maps[image_id].length;
          pixel_index += 4
        ) {
          if (this.activation_maps[image_id][pixel_index] === 255) {
            map_array.push(1);
          } else {
            map_array.push(0);
          }
        }
        corrected_activation_maps[image_id] = map_array;
      }

      const payload = {
        from_scratch: from_scratch,
        activation_maps: corrected_activation_maps
      };

      this.saveData(payload);

      this.dialog = false;

      this.loading = true;
    },
    updateTrainingProgress: function() {
      const path = "http://localhost:5000/training_progress";
      axios
        .get(path)
        .then(res => {
          this.current_epoch = res.data.current_epoch;
          this.total_epochs = res.data.total_epochs;
          this.time_remaining = res.data.time_remaining;

          console.log(this.current_epoch);
          this.loading_message =
            "Current Epoch:" +
            String(this.current_epoch) +
            "/" +
            String(this.total_epochs) +
            " " +
            "Estimated Time Remaining: " +
            this.time_remaining;
          this.check_number = this.check_number + 1;
        })
        .catch(error => {
          console.error(error);
        });
    }
  },
  created() {
    this.getActivationMap();
  },
  mounted() {}
};
</script>

<style scoped>
#container {
  position: relative;
}
.canvas {
  position: absolute;
  cursor: none;
}
#draw {
  position: relative;
  cursor: none;
  float: left;
}
</style>