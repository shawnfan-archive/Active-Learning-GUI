<template>
  <v-app id="main">
    <v-container align-center justify-center>
      <!-- Toolbar -->
      <v-row>
        <v-toolbar dense class="my-2">
          <v-app-bar-nav-icon></v-app-bar-nav-icon>

          <v-toolbar-title>An Active Learning Approach to Acute Stroke Detection</v-toolbar-title>

          <v-spacer></v-spacer>

          <v-toolbar-items>
            <v-btn text v-on:click="toggleSubmissionDialog">
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
                  <v-col cols="2">
                    <!-- Show map -->
                    <!-- <v-col>
                      <v-switch class="mx-4" inset color="indigo" v-model="show_map"></v-switch>
                    </v-col>-->

                    <!-- Undo -->
                    <v-col>
                      <v-btn class="ma-4" fab dark color="purple">
                        <v-icon dark>mdi-undo</v-icon>
                      </v-btn>
                    </v-col>

                    <!-- Paintbrush -->
                    <v-col>
                      <v-menu left offset-y>
                        <template v-slot:activator="{ on }">
                          <v-btn class="ma-4" fab dark color="red" v-on="on">
                            <v-icon dark>mdi-pencil</v-icon>
                          </v-btn>
                        </template>

                        <v-col>
                          <v-btn-toggle>
                            <v-btn
                              v-for="parameter in button_parameters"
                              :key="parameter[0]"
                              fab
                              color="orange"
                              v-on:click="setTool('activate',parameter[0])"
                            >
                              <v-icon :size="parameter[1]">mdi-checkbox-blank-circle</v-icon>
                            </v-btn>
                          </v-btn-toggle>
                        </v-col>
                      </v-menu>
                    </v-col>

                    <!-- Eraser -->
                    <v-col>
                      <v-menu left offset-y>
                        <template v-slot:activator="{ on }">
                          <v-btn class="ma-4" fab dark color="cyan" v-on="on">
                            <v-icon dark>mdi-eraser</v-icon>
                          </v-btn>
                        </template>

                        <v-col>
                          <v-btn-toggle>
                            <v-btn
                              v-for="parameter in button_parameters"
                              :key="parameter[0]"
                              fab
                              color="blue"
                              v-on:click="setTool('deactivate',parameter[0])"
                            >
                              <v-icon :size="parameter[1]">mdi-checkbox-blank-circle</v-icon>
                            </v-btn>
                          </v-btn-toggle>
                        </v-col>

                        <v-col justify="center">
                          <v-btn text v-on:click="clearActivationMap()">
                            Clear All
                            <v-icon>mdi-close</v-icon>
                          </v-btn>
                        </v-col>
                      </v-menu>
                    </v-col>
                  </v-col>

                  <!-- Canvases -->
                  <v-col cols="10">
                    <v-row align="center" justify="center">
                      <h2>Image ID: {{ current_image.image_id }}</h2>

                      <v-divider class="mx-4" vertical></v-divider>

                      <h2>Disease: {{ current_image.disease }}</h2>
                    </v-row>

                    <ContentLoader class="can" v-if="skeleton_loader" :width="600" :height="500"></ContentLoader>

                    <!--canvas for brain image-->
                    <div v-else>
                      <div class="canvas">
                        <canvas
                          ref="image"
                          v-bind:width="canvas_width"
                          v-bind:height="canvas_height"
                        ></canvas>
                      </div>
                      <!-- canvas for cursor -->
                      <div class="canvas">
                        <canvas
                          ref="cursor"
                          v-bind:width="canvas_width"
                          v-bind:height="canvas_height"
                        ></canvas>
                      </div>
                      <!--canvas for activation map-->
                      <div id="draw">
                        <canvas
                          ref="draw"
                          v-bind:width="canvas_width"
                          v-bind:height="canvas_height"
                          v-on:mousemove="correctActivation"
                        ></canvas>
                      </div>
                    </div>
                  </v-col>
                </v-row>
              </v-container>
            </v-card>

            <!-- Histogram -->
            <v-card class="ma-2" min-height="150">
              <v-row>
                <v-col cols="3">
                  <v-row align="center">
                    <v-slider v-model="contrast" prepend-icon="mdi-contrast-circle" min="0" max="300"></v-slider>
                  </v-row>
                  <!-- <v-row justify="center">
                  <v-icon>mdi-contrast-circle</v-icon>
                  </v-row> -->
                </v-col>
                <v-col cols="9">
                  <v-sheet class="ma-1" elevation color="transparent" max-height="75">
                    <v-text>Histogram</v-text>
                    <v-sparkline
                      :key="histogram_key"
                      :value="histogram"
                      height="50"
                      fill
                      auto-draw
                      smooth
                      color='#42b3f4'
                      padding="1"
                    ></v-sparkline>
                  </v-sheet>
                </v-col>
              </v-row>
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
                        :src="require(`../assets/${image.path}`)"
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
            <v-btn color="red darken-1" text v-on:click="toggleSubmissionDialog">Cancel</v-btn>

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
import { ContentLoader } from "vue-content-loader";

export default {
  components: {
    ContentLoader
  },
  data() {
    return {
      current_image: {
        image_id: "",
        disease: "",
        path: ""
      },
      current_image_data: [],
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
      show_map: false,
      button_parameters: [
        [20, 25],
        [10, 12.5],
        [5, 7.5]
      ],
      canvas_width: 600,
      canvas_height: 500,
      graphics: {
        paintbrush: "rgb(254, 0, 0)",
        map_opacity: 0.8
      },
      paint: {
        points: [],
        tool: ""
      },
      skeleton_loader: true,
      skeleton_thumbnail: [1, 2, 3, 4, 5],
      dialog: false,
      model_name: "",
      loading: false,
      contrast: 100,
      check_number: 1,
      current_epoch: null,
      total_epochs: null,
      time_remaining: "",
      loading_message: "Initializing Training...",
      histogram: new Array(256).fill(0),
      histogram_key: 0
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
    },
    contrast: function() {
      let ctx = this.$refs.image.getContext("2d");
      let img = ctx.getImageData(0, 0, this.canvas_width, this.canvas_height);

      let factor = this.contrast / 100;

      for (let i = 0; i < this.current_image_data.length; i += 4) {
        img.data[i + 0] = (this.current_image_data[i + 0] - 127) * factor + 127;
        img.data[i + 1] = (this.current_image_data[i + 1] - 127) * factor + 127;
        img.data[i + 2] = (this.current_image_data[i + 2] - 127) * factor + 127;
      }

      ctx.putImageData(img, 0, 0);

      this.loadHistogram();
    }
  },
  methods: {
    setTool: function(tool, tool_size) {
      this.tool = tool;
      this.tool_size = tool_size;
    },
    getActivationMap: function() {
      //return promise
      return axios
        .get("http://localhost:5000/active_learning")
        .then(res => {
          this.images = res.data.images;
          this.model_name = res.data.latest_model;
          this.activation_maps = res.data.activation_maps;

          this.current_image = this.images[0];
          this.skeleton_loader = false;
        })
        .catch(error => {
          console.error(error);
        });
    },
    loadImage: function(image) {
      //Store current activation map
      this.storeActivationMap();

      //Update current image to new image
      this.current_image = image;
      //Load activation map of new image
      this.loadActivationMap();
      //Load new image onto canvas
      let ctx = this.$refs.image.getContext("2d");
      this.contrast = 100;

      function loadCanvasImage(src) {
        return new Promise(resolve => {
          let canvas_img = new Image();
          canvas_img.src = require(`../assets/${src}`);
          canvas_img.onload = function() {
            ctx.drawImage(canvas_img, 0, 0);
            resolve(canvas_img);
          };
        });
      }

      loadCanvasImage(this.current_image.path).then(canvas_img => {
        let canvas_img_data = ctx.getImageData(
          0,
          0,
          this.canvas_width,
          this.canvas_height
        );
        this.current_image_data = canvas_img_data.data;
        this.loadHistogram();
      });
    },
    loadHistogram: function() {
      this.histogram = new Array(256).fill(0);

      let ctx = this.$refs.image.getContext("2d");
      let canvas_img = ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );
      for (let i = 0; i < canvas_img.data.length; i += 4) {
        this.histogram[canvas_img.data[i]] += 1;
      }
      //Use key change to rerender histogram
      this.histogram_key += 1;
    },
    loadCursor: function(x, y) {
      let ctx = this.$refs.cursor.getContext("2d");
      ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);
      ctx.fillStyle = this.graphics.paintbrush;
      ctx.beginPath();
      ctx.arc(x, y, this.tool_size / 2, 0, 2 * Math.PI);
      ctx.fill();
    },
    loadActivationMap: function() {
      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");
      ctx.strokeStyle = this.graphics.paintbrush;
      ctx.fillStyle = this.graphics.paintbrush;

      //Clear current activation map
      ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);

      ctx.globalCompositeOperation = "source-over";

      let current_activation_map = this.activation_maps[
        this.current_image.image_id
      ];

      for (let row_index = 0; row_index < this.canvas_height; row_index++) {
        for (
          let column_index = 0;
          column_index < this.canvas_width;
          column_index++
        ) {
          if (current_activation_map[row_index][column_index] === 1) {
            ctx.fillRect(column_index, row_index, 1, 1);
          }
        }
      }
    },
    clearActivationMap: function() {
      //Clear current activation map
      let ctx = this.$refs.draw.getContext("2d");
      ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);
    },
    correctActivation: function(event) {
      // load cursor
      this.loadCursor(event.offsetX, event.offsetY);

      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");

      //Set graphics parameters
      ctx.strokeStyle = this.graphics.paintbrush;
      ctx.fillStyle = this.graphics.paintbrush;
      ctx.lineJoin = ctx.lineCap = "round";
      ctx.lineWidth = this.tool_size;

      //Check if using eraser or painbrush
      if (this.tool === "deactivate") {
        ctx.globalCompositeOperation = "destination-out";
      } else {
        ctx.globalCompositeOperation = "source-over";
      }

      canvas.onmousedown = () => {
        //Draw circle
        ctx.beginPath();
        ctx.arc(
          event.offsetX,
          event.offsetY,
          this.tool_size / 2,
          0,
          2 * Math.PI
        );
        ctx.fill();
        //Start drawing
        this.tool_started = true;
        this.paint.points.push({ x: event.offsetX, y: event.offsetY });
        // ctx.beginPath();
        // ctx.moveTo(event.offsetX, event.offsetY);
      };

      if (this.tool_started) {
        this.paint.points.push({ x: event.offsetX, y: event.offsetY });

        ctx.beginPath();

        // ctx.lineTo(event.offsetX, event.offsetY);
        // ctx.stroke();

        let points_length = this.paint.points.length;

        //draw just the last segment
        if (points_length > 1) {
          ctx.moveTo(
            this.paint.points[points_length - 2].x,
            this.paint.points[points_length - 2].y
          );
          ctx.lineTo(
            this.paint.points[points_length - 1].x,
            this.paint.points[points_length - 1].y
          );
        }
        ctx.stroke();
        // ctx.closePath();
      }

      canvas.onmouseup = () => {
        //Stop drawing
        this.tool_started = !this.tool_started;
        //Draw circle
        ctx.beginPath();
        ctx.arc(
          event.offsetX,
          event.offsetY,
          this.tool_size / 2,
          0,
          2 * Math.PI
        );
        ctx.fill();
      };
    },
    storeActivationMap: function() {
    //Store activation map of current image

      //Get RGBA array from canvas
      let map_ctx = this.$refs.draw.getContext("2d");
      let map_data = map_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );

      let current_map = this.activation_maps[this.current_image.image_id];

      //Convert RGBA array into nested binary arrays
      for (
        let pixel_index = 0;
        pixel_index < map_data.data.length;
        pixel_index += 4
      ) {
        let row_index = Math.floor(pixel_index / 4 / this.canvas_width);
        let col_index = (pixel_index / 4) % this.canvas_width;
        if (map_data.data[pixel_index + 3] !== 0) {
          current_map[row_index][col_index] = 1;
        } else {
          current_map[row_index][col_index] = 0;
        }
      }

      this.activation_maps[this.current_image.image_id] = current_map;
      console.log(current_map)
    },
    undoChanges: function() {},
    toggleSubmissionDialog: function() {
      this.dialog = !this.dialog;
    },
    saveData: function(payload) {
      const path = "http://localhost:5000/active_learning";
      axios
        .post(path, payload)
        .then(() => {
          console.log("Stop loading");
          this.loading = false;
          this.getActivationMap()
            .then(returnVal => {
              this.loadActivationMap();
            })
            .catch(err => console.log("Axios err: ", err));
        })
        .catch(error => {
          console.log(error);
          this.getActivationMap();
        });

      setTimeout(this.updateTrainingProgress, 10000);
    },
    onSubmit: function(from_scratch) {
      //Store current activation map
      this.storeActivationMap();
      //Reset loading message
      this.loading_message = "Initializing Training..."; 

      const payload = {
        from_scratch: from_scratch,
        activation_maps: this.activation_maps
      };
      this.saveData(payload);

      //Close submission dialog
      this.dialog = false;

      //Open loading window
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
    this.getActivationMap()
      .then(returnVal => {
        this.loadActivationMap();
        this.loadImage(this.current_image);
      })
      .catch(err => console.log("Axios err: ", err));
  },
  mounted() {}
};
</script>

<style scoped>
.canvas {
  position: absolute;
  cursor: none;
}
#draw {
  cursor: none;
  opacity: 0.3;
}
</style>